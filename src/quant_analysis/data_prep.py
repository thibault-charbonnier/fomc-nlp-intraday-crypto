import pandas as pd
import numpy as np
import ast
from typing import List, Literal, Dict, List

def parse_csv(csv_path: str,
              reactions_mapping: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Parse the enriched sentiment CSV and expand reactions columns.

    Params:
        csv_path: str
            Path to the enriched sentiment CSV file.
        reactions_mapping: Dict[str, int]
            Mapping of symbols to their respective horizons.
            Ex : {"BTCUSDT": [1, 2, ...], "ETHUSDT": [1, 2, ...]}
    """
    df = pd.read_csv(csv_path)

    def _parse_reactions(row: pd.Series, reactions_mapping: Dict[str, List[int]]) -> pd.Series:
        reactions_dict = ast.literal_eval(row["reactions"])

        for symbol, reactions in reactions_dict.items():
            horizon_mapping = reactions_mapping[symbol]
            if len(horizon_mapping) != len(reactions):
                raise ValueError(f"Horizon mapping length does not match reactions length for symbol {symbol}.")
            else:
                for i in range(len(horizon_mapping)):
                    col_name = f"{symbol}_{horizon_mapping[i]}m_bps"
                    row[col_name] = reactions[i]
        return row
    
    df = df.apply(lambda row: _parse_reactions(row, reactions_mapping), axis=1)

    return df.drop(columns=["reactions"])



def load_event_data(
    csv_path: str,
    symbols: List[str],
    windows: List[int],
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Charge et normalise les données d'événements FOMC.

    Le CSV attendu est celui produit par le pipeline complet :
    - 'meeting_date' : str (ex "14-12-2016")
    - 'score_stmt'   : float
    - 'score_qa'     : float
    - 'delta_score'  : float
    - soit des colonnes déjà expansées du type f"{symbol}_{h}m_bps" pour chaque symbol/h
    - soit une colonne 'reactions' sérialisée contenant les réactions marché
      (ex: "{'BTCUSDT': [v1, v2, ...], 'ETHUSDT': {...}}" ou horizon->valeurs)

    Params:
        csv_path : str
            Chemin vers le cache (full_sentiment_cache.csv).
        symbols : List[str]
            Liste de symboles à charger (ex. ["BTCUSDT", "ETHUSDT"]).
        windows : List[int]
            Horizons en minutes pour lesquels on attend les colonnes.
        dayfirst : bool (default True)
            Les dates du pipeline sont "DD-MM-YYYY".

    Returns:
        pd.DataFrame
            meeting_date -> Timestamp UTC trié
            + colonnes de score et de réactions pour tous les symbols/horizons
    """
    df = pd.read_csv(csv_path)

    df["meeting_date"] = pd.to_datetime(
        df["meeting_date"],
        dayfirst=dayfirst,
        utc=True,
        errors="coerce",
    )

    df = df.sort_values("meeting_date").reset_index(drop=True)

    # Colonnes attendues pour tous les symboles et horizons
    expected_cols = [f"{sym}_{h}m_bps" for sym in symbols for h in windows]
    present_expected = [c for c in expected_cols if c in df.columns]

    # Si tout est déjà présent, on renvoie immédiatement.
    if len(present_expected) == len(expected_cols):
        keep_cols = ["meeting_date", "score_stmt", "score_qa", "delta_score"] + expected_cols
        return df[keep_cols].copy()

    # Sinon, on tente de reconstruire depuis 'reactions'
    if "reactions" not in df.columns:
        missing = [c for c in expected_cols if c not in df.columns]
        raise ValueError(
            f"Colonnes manquantes pour {symbols}: {missing}. "
            "Le CSV ne contient pas 'reactions' pour reconstruire ces colonnes."
        )

    # Helper pour parser une cellule 'reactions'
    def _parse_reactions_cell(cell: object):
        # cell peut être NaN, une chaîne dict-like/JSON, ou un dict/list déjà chargé
        if pd.isna(cell):
            return None
        if isinstance(cell, (dict, list, tuple, pd.Series, np.ndarray)):
            return cell
        if isinstance(cell, str):
            try:
                parsed = ast.literal_eval(cell)
            except Exception:
                try:
                    parsed = pd.read_json(cell)
                except Exception:
                    return None
            return parsed
        return None

    # Heuristique pour reconnaître des clés "horizon"
    def _is_horizon_key(k: object) -> bool:
        ks = str(k).lower()
        if ks.endswith("m_bps"):
            ks = ks[:-5]
        if ks.endswith("m"):
            ks = ks[:-1]
        try:
            int(ks)
            return True
        except Exception:
            return False

    # Crée les colonnes vides pour tous les symboles/horizons
    for sym in symbols:
        for h in windows:
            col = f"{sym}_{h}m_bps"
            if col not in df.columns:
                df[col] = np.nan

    # Remplissage ligne à ligne
    for idx, row in df.iterrows():
        parsed = _parse_reactions_cell(row["reactions"])
        if parsed is None:
            continue

        for sym in symbols:
            vals = None

            if isinstance(parsed, dict):
                # Cas classique: dict top-level {symbol -> list/dict}
                if sym in parsed:
                    vals = parsed[sym]
                else:
                    # Cas où le dict est directement {horizon -> valeur}
                    # (on applique alors la même série à tous les symbols)
                    if all(_is_horizon_key(k) for k in parsed.keys()):
                        vals = parsed
            else:
                # list/tuple/Series/ndarray : on l'applique à tous les symbols
                vals = parsed

            if vals is None:
                continue

            if isinstance(vals, dict):
                # On mappe par clés d'horizon
                for h in windows:
                    for key in (h, str(h), f"{h}m", f"{h}m_bps"):
                        if key in vals:
                            try:
                                df.at[idx, f"{sym}_{h}m_bps"] = float(vals[key])
                                break
                            except Exception:
                                # si conversion impossible, on laisse NaN
                                pass
            elif isinstance(vals, (list, tuple, pd.Series, np.ndarray)):
                # On mappe par position, dans l'ordre des windows
                for i, v in enumerate(vals):
                    if i >= len(windows):
                        break
                    try:
                        df.at[idx, f"{sym}_{windows[i]}m_bps"] = float(v)
                    except Exception:
                        df.at[idx, f"{sym}_{windows[i]}m_bps"] = np.nan

    keep_cols = ["meeting_date", "score_stmt", "score_qa", "delta_score"] + expected_cols
    return df[keep_cols].copy()


def add_tone_buckets(
    df: pd.DataFrame,
    score_col: Literal["score_stmt", "score_qa", "delta_score"],
    q: float = 0.25,
    col_name: str = "tone_bucket",
) -> pd.DataFrame:
    """
    Calcule un bucket de 'ton' FOMC à partir d'un score (hawkish/dovish).

    Idée:
        - On prend le quantile q (ex 0.25) et le quantile 1-q (ex 0.75).
        - "dovish"  : score <= q_low
        - "hawkish" : score >= q_high
        - "neutral" : sinon

    Remarques:
        - On suppose qu'on veut une coupure symétrique : q et 1-q.
        - On ne décide pas ici du sens (i.e. plus haut = plus hawkish) :
          on part du principe que ton score_col est déjà orienté
          dans le sens "grand = hawkish, petit = dovish".
          (Si tu inverses le signe du score global, ça inversera les buckets.)

    Params:
        df : pd.DataFrame
            DataFrame sortie de load_event_data().
        score_col : str
            La colonne utilisée pour bucketiser
            ("score_stmt", "score_qa" ou "delta_score").
        q : float (default 0.25)
            Quantile bas. Le quantile haut utilisé sera (1-q).
            Exemple q=0.20 => bas=20%, haut=80%.
        col_name : str (default "tone_bucket")
            Nom de la colonne bucket de sortie.

    Returns:
        pd.DataFrame
            Copie de df avec une nouvelle colonne col_name.
            Valeurs dans { "dovish", "neutral", "hawkish" }.
    """
    df = df.copy()

    q_low = df[score_col].quantile(q)
    q_high = df[score_col].quantile(1.0 - q)

    def _label(v: float) -> str:
        if v <= q_low:
            return "dovish"
        elif v >= q_high:
            return "hawkish"
        else:
            return "neutral"

    df[col_name] = df[score_col].apply(_label)

    return df


import re
from typing import List
import pandas as pd

def build_returns_long_format(
    df: pd.DataFrame,
    windows: List[int],
    tone_col: str = "tone_bucket",
) -> pd.DataFrame:
    """
    Transforme le DF 'wide' (une ligne par FOMC) en DF 'long' (plus pratique plots/stats),
    en autodétectant tous les symboles via les colonnes `SYMBOL_{h}m_bps`.

    Sortie :
        meeting_date, score_stmt, score_qa, delta_score, tone_bucket,
        symbol, horizon_min, return_bps
    """
    df = df.copy()

    base_cols = ["meeting_date", "score_stmt", "score_qa", "delta_score"]
    if tone_col in df.columns:
        base_cols.append(tone_col)

    # Regex pour reconnaître les colonnes de rendement : SYMBOL_{h}m_bps
    rx = re.compile(r"^(?P<symbol>[^_]+)_(?P<h>\d+)m_bps$", re.IGNORECASE)

    # Liste (col, symbol, horizon) filtrée par les windows demandées
    targets = []
    for col in df.columns:
        m = rx.match(col)
        if not m:
            continue
        h = int(m.group("h"))
        if h in windows:
            targets.append((col, m.group("symbol"), h))

    if not targets:
        raise ValueError(
            "Aucune colonne de type 'SYMBOL_{h}m_bps' trouvée pour les horizons demandés."
        )

    long_rows = []
    for _, row in df.iterrows():
        for col, symbol, h in targets:
            if col not in df.columns:
                # ne devrait pas arriver car on vient de scanner df.columns
                continue
            rec = {
                "meeting_date": row["meeting_date"],
                "score_stmt": row["score_stmt"],
                "score_qa": row["score_qa"],
                "delta_score": row["delta_score"],
                "symbol": symbol,
                "horizon_min": h,
                "return_bps": row[col],
            }
            if tone_col in row.index:
                rec[tone_col] = row[tone_col]
            long_rows.append(rec)

    out = pd.DataFrame(long_rows)
    out = out.sort_values(["meeting_date", "symbol", "horizon_min"]).reset_index(drop=True)
    return out


def expand_reactions_csv(
    csv_path: str,
    reactions_horizons: dict,
    col_template: str = "{sym}_{h}m_bps",
    reactions_col: str = "reactions",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Lecture simple du CSV `full_sentiment_cache.csv` et expansion de la colonne
    `reactions` en colonnes par symbole/horizon.

    Params:
        csv_path: chemin vers le CSV (str)
        reactions_horizons: dict mapping symbol -> list of horizons (e.g. {"BTCUSDT":[1,2,5]})
        col_template: template de nommage des colonnes (par défaut "{sym}_{h}m_bps")
        reactions_col: nom de la colonne contenant les réactions sérialisées
        dayfirst: param pour pd.to_datetime

    Retourne:
        pd.DataFrame: DataFrame original enrichi avec les colonnes demandées
    """
    # version simplifiée: on suppose reactions est dict[str, List[float]]
    df = pd.read_csv(csv_path)

    df["meeting_date"] = pd.to_datetime(
        df.get("meeting_date", None), dayfirst=dayfirst, utc=True, errors="coerce"
    )

    # créer les colonnes vides demandées
    for sym, horizons in reactions_horizons.items():
        for h in horizons:
            col = col_template.format(sym=sym, h=h)
            if col not in df.columns:
                df[col] = np.nan

    # parser très simple: si string -> ast.literal_eval, sinon on espère un dict
    def _parse_simple(cell: object):
        if pd.isna(cell):
            return None
        if isinstance(cell, str):
            try:
                return ast.literal_eval(cell)
            except Exception:
                return None
        return cell

    for idx, row in df.iterrows():
        parsed = _parse_simple(row.get(reactions_col, None))
        if not isinstance(parsed, dict):
            continue

        for sym, horizons in reactions_horizons.items():
            vals = parsed.get(sym, None)
            if vals is None:
                continue
            # on attend une liste/tuple de valeurs dans l'ordre des horizons
            if not isinstance(vals, (list, tuple, np.ndarray)):
                continue
            for i, v in enumerate(vals):
                if i >= len(horizons):
                    break
                h = horizons[i]
                col = col_template.format(sym=sym, h=h)
                try:
                    df.at[idx, col] = float(v)
                except Exception:
                    df.at[idx, col] = np.nan

    return df
