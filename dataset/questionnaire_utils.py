import numpy as np
import pandas as pd

# source code by Szymon Bobek: https://colab.research.google.com/drive/1Hj6yH4UIrAp1Jp6B1U542vcdSkXuZHUd (accessed and modified: 5 May 2026)
def make_counterfactual_delta_table(
    query_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    decimals: int = 3,
    tol: float = 1e-12,
    cf_index_prefix: str = "CF #",
    feature_types:dict[str,str]=None
):
    """
    Build a styled DataFrame indicating how much each feature should change
    to obtain each counterfactual, with signed formatting like +0.232.

    Parameters
    ----------
    query_df : pd.DataFrame
        The original instance(s), unscaled. If it contains 1 row, it will be broadcast
        to the number of rows in cf_df.
    cf_df : pd.DataFrame
        The generated counterfactual instance(s), unscaled. Must have the same columns as query_df.
        Each row is a distinct counterfactual.
    decimals : int
        Number of decimal places to display for deltas.
    tol : float
        Absolute tolerance under which a delta is considered zero (displayed as blank).
    cf_index_prefix : str
        Prefix for counterfactual row names.

    Returns
    -------
    styled : pd.io.formats.style.Styler
        A styled table with signed deltas and color cues (green for increases, red for decreases).
    delta_df : pd.DataFrame
        The numeric delta DataFrame (cf_df - query_df).
    """

    # --- Basic validation
    if not isinstance(query_df, pd.DataFrame) or not isinstance(cf_df, pd.DataFrame):
        raise TypeError("query_df and cf_df must be pandas DataFrames")

    if list(query_df.columns) != list(cf_df.columns):
        raise ValueError("query_df and cf_df must have identical columns (same order).")

    # Broadcast query row if a single instance is provided
    if len(query_df) == 1 and len(cf_df) > 1:
        query_aligned = pd.DataFrame(
            np.repeat(query_df.values, repeats=len(cf_df), axis=0),
            columns=query_df.columns,
            index=cf_df.index
        )
    else:
        # Otherwise, they must have the same number of rows
        if len(query_df) != len(cf_df):
            raise ValueError(
                f"Row mismatch: query_df has {len(query_df)} rows, cf_df has {len(cf_df)} rows. "
                f"Provide one query row to broadcast or match the counts."
            )
        query_aligned = query_df.copy()
        query_aligned.index = cf_df.index

    # Compute deltas (what to add to the original to reach the CF)
    delta = cf_df - query_aligned

    # Index: label rows as CF #1, CF #2, ...
    delta.index = [f"{cf_index_prefix}{i+1}" for i in range(len(delta))]

    # Format as signed strings (blank if ~0)
    def _fmt_signed(x,col_name:str):
        if pd.isna(x):
            return ""
        if abs(x) < tol:
            return ""
        if feature_types is not None and col_name in feature_types.keys() \
            and (feature_types[str(col_name)] == "ordinal" or feature_types[str(col_name)] == "nominal"):
            s = f"{int(x):+d}"
        else:
            s = f"{x:+.{decimals}f}"
        return s

    formatted = delta.apply(lambda col: col.map(lambda val: _fmt_signed(val,col.name)))

    # Styling: green for positive, red for negative, gray for zero/blank
    def color_changes(val):
        if val == "":
            return "color: #888888;"  # gray for (near-)zero
        return "color: #2e7d32;" if val.strip().startswith("+") else "color: #c62828;"

    styled = (
        formatted.style
        .applymap(color_changes)
        .set_properties(**{
            "font-family": "Segoe UI, Roboto, Arial, sans-serif",
            "font-size": "12.5px",
            "white-space": "nowrap",
        })
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "600"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 10px")]},
        ])
        .set_caption("Required feature adjustments to obtain each counterfactual (unscaled)")
    )

    return styled, delta


def counterfactual_instructions(delta_df: pd.DataFrame, tol: float = 1e-12, decimals: int = 3, feature_types: dict[str,str] = None):
    """
    Build natural-language instructions for each counterfactual row:
    Example: 'Increase X by 0.232; decrease Y by 1.50'

    Returns
    -------
    dict: {row_label: instruction_str}
    """
    instructions = {}
    for idx, row in delta_df.iterrows():
        parts = []
        for col, v in row.items():
            if pd.isna(v) or abs(v) < tol:
                continue
            direction = "increase" if v > 0 else "decrease"
            if feature_types is not None and col in feature_types.keys() and (feature_types[str(col)] == "ordinal" or feature_types[str(col)] == "nominal"):
                parts.append(f"{direction} {col} by {abs(int(v))}")
            else:
                parts.append(f"{direction} {col} by {abs(v):.{decimals}f}")
        instructions[idx] = "; ".join(parts) if parts else "No changes needed"
    return instructions