import re
import time
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from bs4 import BeautifulSoup

from ..tools.logging_config import get_logger

logger = get_logger(__name__)


# ---------- Data container ----------
@dataclass
class FOMCMeeting:
    """
    One FOMC meeting reference.

    Params:
        date_str : str
            Meeting date as YYYYMMDD (Fed timestamp).
        url : str
            URL of the 'fomcpresconfYYYYMMDD.htm' page (press conf page).
    """
    date_str: str
    url: str


# ---------- Downloader ----------
class FOMCDownloader:
    """
    Downloader for FOMC material (statement + press conf transcript) from federalreserve.gov.
    Handles modern PDFs and legacy HTML statements as fallback.
    """

    def __init__(self,
                 start_year: int,
                 end_year: int,
                 out_dir: str = "data/_cache/fomc_data",
                 sleep_sec: float = 0.5) -> None:
        """
        Params:
            start_year : int
                First year to download (inclusive).
            end_year : int
                Last year to download (inclusive).
            out_dir : str
                Output base directory.
            sleep_sec : float
                Pause between requests.
        """
        self.start_year = start_year
        self.end_year = end_year
        self.base_root = "https://www.federalreserve.gov"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_sec = sleep_sec

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; FOMCDownloader/1.0)"
        })

    # -------------------- Public --------------------
    def sync(self) -> None:
        """
        For each year:
            - discover meetings
            - create folder data/_cache/fomc_data/<YYYYMMDD>/
            - download statement (PDF or HTML fallback) + pressconf (PDF)
        """
        logger.info("Starting FOMC historical data sync...")
        for year in range(self.start_year, self.end_year + 1):
            meetings = self._discover_meetings_for_year(year)
            logger.info("Found %d meetings in %d", len(meetings), year)

            for mtg in meetings:
                logger.info("    Processing meeting %s", mtg.date_str)
                mdir = (self.out_dir / mtg.date_str)
                mdir.mkdir(parents=True, exist_ok=True)

                # Statement: try PDF, else legacy HTML
                self._download_statement_any(
                    date_str=mtg.date_str,
                    meeting_dir=mdir,
                    meeting_page_url=mtg.url
                )

                # Press conference (PDF)
                self._download_pressconf_pdf(
                    date_str=mtg.date_str,
                    meeting_dir=mdir,
                    meeting_page_url=mtg.url
                )

                time.sleep(self.sleep_sec)

        logger.info("Sync complete.")

    # -------------------- Discovery --------------------
    def _discover_meetings_for_year(self, year: int) -> List[FOMCMeeting]:
        """
        Find 'fomcpresconfYYYYMMDD.htm' links for a given year.

        Order:
          1) /monetarypolicy/fomchistorical{YEAR}.htm
          2) /monetarypolicy/fomccalendars.htm (fallback)
        """
        logger.info("Discovering meetings for year %d", year)

        candidates = [
            f"{self.base_root}/monetarypolicy/fomchistorical{year}.htm",
            f"{self.base_root}/monetarypolicy/fomccalendars.htm",
        ]

        html = None
        for url in candidates:
            try:
                r = self.session.get(url, timeout=15)
            except Exception:
                continue
            if r.status_code == 200 and "fomc" in r.text.lower():
                html = r.text
                break

        if html is None:
            logger.warning("Could not fetch calendar for %d", year)
            return []

        soup = BeautifulSoup(html, "html.parser")

        meetings: List[FOMCMeeting] = []
        pat = re.compile(r"fomcpresconf(\d{8})\.htm", re.IGNORECASE)

        for a in soup.find_all("a", href=True):
            m = pat.search(a["href"])
            if not m:
                continue
            date_str = m.group(1)
            if not date_str.startswith(str(year)):
                continue
            full = a["href"]
            if not full.startswith("http"):
                full = self.base_root + full
            meetings.append(FOMCMeeting(date_str=date_str, url=full))

        # de-dupe + sort
        uniq = {m.date_str: m for m in meetings}
        out = [uniq[k] for k in sorted(uniq.keys())]

        logger.info("Found %d meetings in %d", len(out), year)
        return out

    # -------------------- Statement (PDF or HTML) --------------------
    def _download_statement_any(self,
                                date_str: str,
                                meeting_dir: Path,
                                meeting_page_url: str) -> bool:
        """
        Try to download the FOMC policy statement.

        Priority:
          1) Modern direct PDF under /monetarypolicy/files/
          2) Legacy pressrelease PDF/HTML under /newsevents/pressreleases/
          3) Scrape meeting page for monetary{DATE}a*.pdf|htm|html
        Saves:
          - statement.pdf when PDF is found
          - statement.html when only HTML exists
        """
        # 1) Modern direct PDFs
        modern_pdfs = [
            f"{self.base_root}/monetarypolicy/files/monetary{date_str}a1.pdf",
            f"{self.base_root}/monetarypolicy/files/monetary{date_str}a.pdf",
        ]
        for url in modern_pdfs:
            if self._try_download_pdf(url, meeting_dir / "statement.pdf"):
                logger.info("        Downloaded statement.pdf")
                return True

        # 2) Legacy pressrelease paths
        legacy_roots = [
            f"{self.base_root}/newsevents/pressreleases/monetary{date_str}a1",
            f"{self.base_root}/newsevents/pressreleases/monetary{date_str}a",
        ]
        # 2a) legacy PDFs
        for base in legacy_roots:
            if self._try_download_pdf(base + ".pdf", meeting_dir / "statement.pdf"):
                logger.info("        Downloaded statement.pdf")
                return True
        # 2b) legacy HTMLs
        for base in legacy_roots:
            if self._try_download_html(base + ".htm", meeting_dir / "statement.html"):
                logger.info("        Downloaded statement.html")
                return True
            if self._try_download_html(base + ".html", meeting_dir / "statement.html"):
                logger.info("        Downloaded statement.html")
                return True

        # 3) Scrape meeting page for any monetary{date}a*.pdf|htm|html
        try:
            resp = self.session.get(meeting_page_url, timeout=15)
        except Exception:
            resp = None
        if resp is not None and resp.status_code == 200:
            stmt_url = self._extract_statement_url_any(resp.text, date_str)
            if stmt_url:
                if not stmt_url.startswith("http"):
                    stmt_url = self.base_root + stmt_url
                if stmt_url.lower().endswith(".pdf"):
                    if self._try_download_pdf(stmt_url, meeting_dir / "statement.pdf"):
                        logger.info("        Downloaded statement.pdf")
                        return True
                else:
                    if self._try_download_html(stmt_url, meeting_dir / "statement.html"):
                        logger.info("        Downloaded statement.html")
                        return True

        logger.warning("        Could not find ANY statement file for %s", date_str)
        return False

    # -------------------- Press conf (PDF) --------------------
    def _download_pressconf_pdf(self,
                                date_str: str,
                                meeting_dir: Path,
                                meeting_page_url: str) -> bool:
        """
        Download press conference transcript (PDF). Scrape fallback if needed.
        """
        direct = f"{self.base_root}/monetarypolicy/files/FOMCpresconf{date_str}.pdf"
        if self._try_download_pdf(direct, meeting_dir / "pressconf.pdf"):
            logger.info("        Downloaded pressconf.pdf")
            return True

        try:
            resp = self.session.get(meeting_page_url, timeout=15)
        except Exception:
            resp = None
        if resp is not None and resp.status_code == 200:
            pc_url = self._extract_pressconf_pdf_url(resp.text, date_str)
            if pc_url:
                if not pc_url.startswith("http"):
                    pc_url = self.base_root + pc_url
                if self._try_download_pdf(pc_url, meeting_dir / "pressconf.pdf"):
                    logger.info("        Downloaded pressconf.pdf")
                    return True

        logger.warning("        Could not download any press conference transcript for this meeting.")
        return False

    # -------------------- Low-level fetchers --------------------
    def _try_download_pdf(self, url: str, out_path: Path) -> bool:
        """GET url and write it as PDF if HTTP 200 and header/content looks like PDF."""
        try:
            r = self.session.get(url, timeout=20)
        except Exception as e:
            logger.debug("Request failed %s: %s", url, e)
            return False

        if r.status_code == 200 and (b"%PDF" in r.content[:10] or
                                     "application/pdf" in r.headers.get("Content-Type", "").lower()):
            out_path.write_bytes(r.content)
            return True

        logger.debug("No PDF at %s (status %s)", url, getattr(r, "status_code", "NA"))
        return False

    def _try_download_html(self, url: str, out_path: Path) -> bool:
        """GET url and save raw HTML if HTTP 200 and content looks HTML."""
        try:
            r = self.session.get(url, timeout=20)
        except Exception as e:
            logger.debug("Request failed %s: %s", url, e)
            return False

        ctype = r.headers.get("Content-Type", "").lower()
        looks_html = ("text/html" in ctype) or (r.content[:200].lower().startswith(b"<!doctype html")
                                                or b"<html" in r.content[:300].lower())

        if r.status_code == 200 and looks_html:
            out_path.write_bytes(r.content)
            return True

        logger.debug("No HTML at %s (status %s, ctype %s)", url, getattr(r, "status_code", "NA"), ctype)
        return False

    # -------------------- Parsers --------------------
    @staticmethod
    def _extract_statement_url_any(html: str, date_str: str) -> Optional[str]:
        """
        Find a link to statement:
          monetary{DATE}a[digits].pdf | .htm | .html
        """
        soup = BeautifulSoup(html, "html.parser")
        pat = re.compile(rf"monetary{date_str}a\d*\.(pdf|htm|html)$", re.IGNORECASE)
        for a in soup.find_all("a", href=True):
            if pat.search(a["href"]):
                return a["href"]
        return None

    @staticmethod
    def _extract_pressconf_pdf_url(html: str, date_str: str) -> Optional[str]:
        """Find FOMCpresconf{DATE}.pdf on the meeting page."""
        soup = BeautifulSoup(html, "html.parser")
        pat = re.compile(rf"FOMCpresconf{date_str}\.pdf", re.IGNORECASE)
        for a in soup.find_all("a", href=True):
            if pat.search(a["href"]):
                return a["href"]
        return None
