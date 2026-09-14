import re
import numpy as np
import pandas as pd

ENCODE_MAP = {
    "Strongly Agree": 2,
    "Agree": 1,
    "Neutral": 0,
    "Disagree": -1,
    "Strongly Disagree": -2,
    "No Comments": np.nan,
    "": np.nan,
}

ENGAGEMENT_THRESHOLD = 0.5  # drop respondents who answered less than this fraction


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and set the response ID as the index."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df.index.name = "respondent_id"
    return df


def encode_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Map Likert text responses to numeric -2..+2, NaN for missing/no-comment."""
    encoded = df.apply(lambda col: col.map(ENCODE_MAP))
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    return encoded


def build_statement_metadata(columns) -> pd.DataFrame:
    """
    Parse statement columns like 'T01. Artificial Intelligence will improve...'
    into (statement_id, category, full_text).
    """
    records = []
    for col in columns:
        m = re.match(r"^([A-Za-z]+)(\d+)\.\s*(.*)$", str(col))
        if m:
            prefix, num, text = m.groups()
            records.append(
                {
                    "statement_id": f"{prefix}{num}",
                    "category": prefix,
                    "full_text": text.strip(),
                    "raw_column": col,
                }
            )
    return pd.DataFrame(records).set_index("statement_id")


def compute_engagement(encoded: pd.DataFrame) -> pd.Series:
    """Fraction of statements each respondent actually answered (non-missing)."""
    return 1 - encoded.isna().sum(axis=1) / encoded.shape[1]


def filter_low_engagement(
    encoded: pd.DataFrame, threshold: float = ENGAGEMENT_THRESHOLD
):
    """Drop respondents below the engagement threshold. Returns (filtered_df, dropped_ids)."""
    engagement = compute_engagement(encoded)
    keep_mask = engagement >= threshold
    dropped_ids = encoded.index[~keep_mask].tolist()
    return encoded.loc[keep_mask].copy(), dropped_ids, engagement


def std_rescale_per_respondent(encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale each respondent's row by their own std (NO mean subtraction).
    This corrects for extreme-responder vs. mild-responder scale usage while
    preserving genuine uniform-strong opinions (unlike full z-scoring, which
    would incorrectly pull a genuine all-Strongly-Agree respondent to 0).
    """
    row_std = encoded.std(axis=1, skipna=True)
    return encoded.div(row_std, axis=0)


def run_pipeline(csv_path: str):
    """Full pipeline entry point. Returns a dict of all artifacts."""
    raw = load_raw(csv_path)
    encoded = encode_responses(raw)
    statement_meta = build_statement_metadata(raw.columns)

    # rename columns to short statement IDs (T01, E01, ...) for readability downstream
    col_map = {row["raw_column"]: sid for sid, row in statement_meta.iterrows()}
    encoded = encoded.rename(columns=col_map)

    filtered, dropped_ids, engagement = filter_low_engagement(encoded)
    rescaled = std_rescale_per_respondent(filtered)

    print(f"Loaded {len(raw)} respondents, {len(statement_meta)} statements.")
    print(f"Dropped {len(dropped_ids)} respondents below "
          f"{ENGAGEMENT_THRESHOLD:.0%} engagement: {dropped_ids}")
    print(f"Retained {len(filtered)} respondents for network construction.")

    return {
        "response_matrix_raw": filtered,        # encoded -2..+2, NaN, engagement-filtered
        "response_matrix_rescaled": rescaled,    # std-rescaled, for User-User network
        "statement_metadata": statement_meta,
        "engagement_scores": engagement,
        "dropped_respondents": dropped_ids,
    }


if __name__ == "__main__":
    artifacts = run_pipeline("../Survey_Results_UC.csv")
    artifacts["response_matrix_raw"].to_csv("../processed/response_matrix.csv")
    artifacts["statement_metadata"].to_csv("../processed/statement_metadata.csv")