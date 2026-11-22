import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Representative Results", page_icon="📊", layout="wide")
st.title("Representative Results")
st.write("View the representative roster, capacity, and download the cleaned CSV.")

# Reuse the representatives CSV uploaded on the home page when available.
rep_file = st.session_state.get("rep_file")
if rep_file is None:
    rep_file = st.file_uploader("Upload representatives CSV", type=["csv"], key="representatives_page_upload")

if rep_file is None:
    st.info("Upload the representatives CSV on the home page, or upload it here to view this page.")
    st.stop()

try:
    reps = pd.read_csv(io.BytesIO(rep_file.getvalue()), encoding="utf-8-sig")
    reps.columns = [str(column).replace("\ufeff", "").strip().replace(" ", "_") for column in reps.columns]
    aliases = {
        "Senority_Level": "Seniority_Level",
        "Serniority_Level": "Seniority_Level",
        "Senioriy_Level": "Seniority_Level",
    }
    reps = reps.rename(columns={column: aliases.get(column, column) for column in reps.columns})

    required = {
        "Rep_ID", "Name", "Region", "Language",
        "Industry_Vertical_Expertise", "Current_Capacity", "Max_Capacity",
    }
    missing = required - set(reps.columns)
    if missing:
        st.error("Representatives CSV is missing required columns: " + ", ".join(sorted(missing)))
        st.stop()

    if "Seniority_Level" not in reps.columns:
        reps["Seniority_Level"] = "unknown"
    reps["Seniority_Level"] = reps["Seniority_Level"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    for column in ("Current_Capacity", "Max_Capacity"):
        reps[column] = pd.to_numeric(reps[column], errors="coerce").fillna(0).clip(lower=0)

    reps["Available_Capacity"] = (reps["Max_Capacity"] - reps["Current_Capacity"]).clip(lower=0)
    reps["Capacity_Utilization"] = (
        reps["Current_Capacity"]
        .div(reps["Max_Capacity"].where(reps["Max_Capacity"] > 0))
        .fillna(1.0)
        .clip(lower=0)
    )

    metrics = st.columns(3)
    metrics[0].metric("Representatives", f"{len(reps):,}")
    metrics[1].metric("Available capacity", f"{reps['Available_Capacity'].sum():,.0f}")
    utilization = reps["Capacity_Utilization"].mean() if not reps.empty else 0
    metrics[2].metric("Average utilization", f"{utilization:.1%}")

    display = reps.copy()
    display["Capacity_Utilization"] = (display["Capacity_Utilization"] * 100).round(1).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download representative results",
        data=reps.to_csv(index=False).encode("utf-8-sig"),
        file_name="representative_results.csv",
        mime="text/csv",
    )
except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not read the representatives CSV: {error}")
