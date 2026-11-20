from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Lead Matcher", page_icon="🎯", layout="wide")

st.title("🎯 Lead Matcher")
st.markdown(
    "Upload a lead list, describe what a good-fit prospect looks like, and get a ranked, "
    "downloadable shortlist. Your data is processed in this app session."
)


def as_display(value: Any) -> str:
    """Make categories selectable even when a column contains mixed value types."""
    return str(value)


def numeric_score(values: pd.Series, low: float, high: float) -> pd.Series:
    """Score values from 0 to 100 by closeness to the selected target interval."""
    numbers = pd.to_numeric(values, errors="coerce")
    if low > high:
        low, high = high, low
    result = pd.Series(0.0, index=values.index)
    inside = numbers.between(low, high)
    result.loc[inside] = 100.0
    below = numbers < low
    above = numbers > high
    if below.any():
        distance = (low - numbers.loc[below]).abs()
        scale = max(abs(high - low), abs(low), 1.0)
        result.loc[below] = (100.0 * (1.0 - distance / scale)).clip(0, 100)
    if above.any():
        distance = (numbers.loc[above] - high).abs()
        scale = max(abs(high - low), abs(high), 1.0)
        result.loc[above] = (100.0 * (1.0 - distance / scale)).clip(0, 100)
    return result.fillna(0.0)


def make_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.header("1. Upload leads")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    st.caption("Tip: include a header row and one lead per row.")

if uploaded is None:
    st.info("Start by uploading a CSV file of leads.")
    with st.expander("What columns can I use?"):
        st.write(
            "Any columns are supported. For example: company, contact, industry, "
            "location, employee_count, annual_revenue, or technology. Choose the fields "
            "that matter and set a target for each one after uploading."
        )
    st.stop()

try:
    leads = pd.read_csv(uploaded)
except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if leads.empty:
    st.error("This CSV has headers but no lead rows.")
    st.stop()
if len(leads.columns) == 0:
    st.error("No columns were found. Check that the CSV has a header row.")
    st.stop()

# Normalize column labels to strings for consistent widgets and export.
leads.columns = [str(column) for column in leads.columns]
leads = leads.loc[:, ~leads.columns.duplicated()]

st.success(f"Loaded {len(leads):,} leads across {len(leads.columns)} columns.")
with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(leads.head(10), use_container_width=True, hide_index=True)

st.header("2. Define your ideal lead")
all_columns = list(leads.columns)
selected = st.multiselect(
    "Which columns should influence the match score?",
    options=all_columns,
    default=all_columns[: min(3, len(all_columns))],
    help="Contact details and identifiers need not be selected unless they help qualify a lead.",
)

if not selected:
    st.warning("Select at least one column to calculate match scores.")
    st.stop()

settings: dict[str, dict[str, Any]] = {}
for index, column in enumerate(selected):
    series = leads[column]
    non_empty = series.dropna()
    numeric = pd.to_numeric(non_empty, errors="coerce")
    is_numeric = len(non_empty) > 0 and numeric.notna().mean() >= 0.8
    with st.expander(f"{column} — {'numeric' if is_numeric else 'category'}", expanded=index == 0):
        weight = st.slider(
            f"Importance of {column}", 1, 10, 5, key=f"weight_{index}",
            help="Higher importance gives this field more influence on the final score.",
        )
        if is_numeric:
            vals = numeric.dropna()
            minimum, maximum = float(vals.min()), float(vals.max())
            if minimum == maximum:
                st.caption(f"All available values are {minimum:g}.")
                low, high = minimum, maximum
            else:
                low, high = st.slider(
                    f"Ideal {column} range", min_value=minimum, max_value=maximum,
                    value=(minimum, maximum), key=f"range_{index}",
                    help="Values inside this range score 100%; values farther away score lower.",
                )
            settings[column] = {"kind": "numeric", "weight": weight, "low": low, "high": high}
        else:
            categories = sorted({as_display(value) for value in non_empty})
            if not categories:
                st.caption("This column has no values to match.")
                preferred: list[str] = []
            else:
                preferred = st.multiselect(
                    f"Preferred {column} values", options=categories, default=categories,
                    key=f"categories_{index}",
                    help="Selected values score 100%. Unselected values score 0%.",
                )
            settings[column] = {"kind": "category", "weight": weight, "preferred": preferred}

st.header("3. Ranked matches")
scored = leads.copy()
weighted_total = pd.Series(0.0, index=scored.index)
weight_total = 0
score_columns: dict[str, str] = {}

for index, (column, config) in enumerate(settings.items()):
    if config["kind"] == "numeric":
        component = numeric_score(scored[column], config["low"], config["high"])
    else:
        preferred = set(config["preferred"])
        component = scored[column].map(
            lambda value: 100.0 if pd.notna(value) and as_display(value) in preferred else 0.0
        )
    score_name = f"{column} match"
    # Avoid collisions with source columns or other generated columns.
    while score_name in scored.columns:
        score_name = f"{score_name}_"
    scored[score_name] = component.round(1)
    score_columns[column] = score_name
    weighted_total += component * config["weight"]
    weight_total += config["weight"]

scored["Match score"] = (weighted_total / weight_total).round(1) if weight_total else 0.0
scored["Match tier"] = pd.cut(
    scored["Match score"], bins=[-1, 39.9, 69.9, 100],
    labels=["Low", "Good", "Excellent"],
).astype("string")
scored = scored.sort_values("Match score", ascending=False, kind="stable")

first, second, third = st.columns(3)
first.metric("Leads scored", f"{len(scored):,}")
second.metric("Excellent matches", f"{(scored['Match tier'] == 'Excellent').sum():,}")
third.metric("Average score", f"{scored['Match score'].mean():.1f} / 100")

minimum_score = st.slider("Minimum match score", 0, 100, 0)
shortlist = scored.loc[scored["Match score"] >= minimum_score]
st.caption(f"Showing {len(shortlist):,} of {len(scored):,} leads, ranked by match score.")
st.dataframe(shortlist, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download ranked leads as CSV",
    data=make_csv(shortlist),
    file_name="ranked_leads.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("How scoring works"):
    st.markdown(
        "- A categorical field scores **100** when its value is selected as preferred, otherwise **0**.\n"
        "- A numeric field scores **100** inside the ideal range and decreases with distance outside it.\n"
        "- The overall score is the weighted average of the selected field scores.\n"
        "- Match tiers are **Excellent** (70–100), **Good** (40–69.9), and **Low** (below 40)."
    )
