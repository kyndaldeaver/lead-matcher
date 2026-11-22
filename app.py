import io
import re

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Team 36's Lead Matcher System",
    page_icon="📊",
    layout="wide",
)

LEAD_COLUMNS = {
    "Lead_ID", "Company_Name", "Region", "Language", "Industry_Vertical",
    "Company_Size", "Recent_Activity", "VIP_Referral",
}
# Seniority_Level is optional: blank or absent values are treated as unknown.
REP_COLUMNS = {
    "Rep_ID", "Name", "Region", "Language", "Industry_Vertical_Expertise",
    "Current_Capacity", "Max_Capacity",
}

REGION_MAP = {
    "asapacific": "asia pacific", "asiapacific": "asia pacific", "apac": "asia pacific",
    "europ": "europe", "europe": "europe", "eur": "europe",
    "northamerica": "north america", "nortamer": "north america",
    "nortamerica": "north america", "northamer": "north america",
    "asiapacificeurope": "asia pacific and europe",
    "northamericaasiapacific": "north america and asia pacific",
}
LANGUAGE_MAP = {
    "en": "english", "english": "english", "es": "spanish", "spanish": "spanish",
    "de": "german", "german": "german",
}
ACTIVITY_MAP = {
    "subscribed": "subscribed", "subscribedtoblog": "subscribed to blog",
    "downloadedwhitepaper": "downloaded whitepaper", "requestedlivedemo": "requested live demo",
    "openedmarketingemail": "opened marketing email", "contactsales": "contact sales",
    "visitedpricingpage": "visited pricing page", "startedfreetrial": "started free trial",
    "attendedwebinar": "attended webinar", "unknown": "unknown", "na": "unknown",
}
SENIORITY_MAP = {
    "mgr": "manager", "manager": "manager", "sr": "senior", "senior": "senior",
    "jr": "junior", "junior": "junior",
}


def normalize_headers(frame):
    """Trim header whitespace/BOM and accept the common Seniority misspelling."""
    frame = frame.copy()
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    aliases = {"Senority_Level": "Seniority_Level", "Seniority Level": "Seniority_Level"}
    frame = frame.rename(columns={column: aliases.get(column, column) for column in frame.columns})
    return frame


def require_columns(dataframe, required_columns, file_name):
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{file_name} is missing the following columns: {', '.join(sorted(missing))}")


def normalize_text(series):
    return (
        series.astype("string").str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "n/a": pd.NA, "na": pd.NA})
    )


def normalize_key(series):
    return normalize_text(series).str.replace(r"[^a-z0-9]+", "", regex=True)


def map_normalized(series, mapping):
    return normalize_key(series).map(mapping).astype("string")


def dataframe_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def clean_company_size(series):
    text = normalize_text(series).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    number = pd.to_numeric(text.str.extract(r"([0-9]*\.?[0-9]+)", expand=False), errors="coerce")
    return (number * np.where(text.str.endswith("k", na=False), 1000, 1)).round().astype("Int64")


def regions_overlap(region_one, region_two):
    if pd.isna(region_one) or pd.isna(region_two):
        return False
    one = {item.strip() for item in str(region_one).split(" and ") if item.strip()}
    two = {item.strip() for item in str(region_two).split(" and ") if item.strip()}
    return bool(one.intersection(two))


def clean_leads(leads):
    leads = normalize_headers(leads)
    require_columns(leads, LEAD_COLUMNS, "The leads file")
    leads = leads.copy()
    leads["Lead_ID"] = leads["Lead_ID"].astype("string").str.strip()
    leads["Company_Name"] = leads["Company_Name"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    leads["Region"] = map_normalized(leads["Region"], REGION_MAP)
    leads["Language"] = map_normalized(leads["Language"], LANGUAGE_MAP)
    leads["Industry_Vertical"] = normalize_text(leads["Industry_Vertical"])
    leads["Company_Size"] = clean_company_size(leads["Company_Size"])
    leads["Recent_Activity"] = map_normalized(leads["Recent_Activity"], ACTIVITY_MAP).fillna("unknown")
    vip = normalize_text(leads["VIP_Referral"])
    leads["VIP_Referral"] = vip.map({"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}).astype("boolean")
    return leads


def clean_reps(reps):
    reps = normalize_headers(reps)
    require_columns(reps, REP_COLUMNS, "The representatives file")
    reps = reps.copy()
    if "Seniority_Level" not in reps.columns:
        reps["Seniority_Level"] = "unknown"
    reps["Rep_ID"] = reps["Rep_ID"].astype("string").str.strip()
    reps["Name"] = (
        reps["Name"].astype("string")
        .str.replace(r"\s*\([^)]*\)\s*", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    reps["Region"] = map_normalized(reps["Region"], REGION_MAP)
    reps["Language"] = map_normalized(reps["Language"], LANGUAGE_MAP)
    reps["Industry_Vertical_Expertise"] = normalize_text(reps["Industry_Vertical_Expertise"])
    reps["Seniority_Level"] = map_normalized(reps["Seniority_Level"], SENIORITY_MAP).fillna("unknown")
    for column in ("Current_Capacity", "Max_Capacity"):
        reps[column] = pd.to_numeric(reps[column], errors="coerce").fillna(0).clip(lower=0).round().astype("Int64")
    current = reps["Current_Capacity"].astype("Float64")
    maximum = reps["Max_Capacity"].astype("Float64")
    reps["Capacity_Utilization"] = current.div(maximum.where(maximum > 0)).fillna(1.0).clip(lower=0)
    reps["Available_Capacity"] = (reps["Max_Capacity"] - reps["Current_Capacity"]).clip(lower=0)
    return reps


def score_leads(leads):
    leads = leads.copy()
    size_score = np.select(
        [leads["Company_Size"].ge(1000).fillna(False), leads["Company_Size"].ge(100).fillna(False), leads["Company_Size"].isna()],
        [50, 30, 20], default=0,
    )
    activity_points = {
        "requested live demo": 35, "contact sales": 35, "attended webinar": 25,
        "started free trial": 25, "downloaded whitepaper": 15,
        "visited pricing page": 10, "subscribed to blog": 5, "subscribed": 5,
    }
    industry_points = {"technology": 15, "healthcare": 10, "finance": 10}
    score = size_score + leads["Recent_Activity"].map(activity_points).fillna(0).astype(int)
    score += leads["Industry_Vertical"].map(industry_points).fillna(0).astype(int)
    is_vip = leads["VIP_Referral"].fillna(False).astype(bool)
    leads["Score"] = pd.Series(score, index=leads.index).mask(is_vip, 100)
    leads["Priority"] = np.select(
        [leads["Score"].ge(85), leads["Score"].ge(60), leads["Score"].ge(40)],
        ["critical", "high", "medium"], default="low",
    )
    return leads.sort_values(["Score", "Lead_ID"], ascending=[False, True], kind="stable").reset_index(drop=True)


def validate_data(leads, reps):
    warnings = []
    if leads["Lead_ID"].isna().any():
        warnings.append("Some leads have missing Lead_ID values.")
    duplicate_leads = int(leads["Lead_ID"].duplicated().sum())
    if duplicate_leads:
        warnings.append(f"{duplicate_leads} duplicate Lead_ID values were found.")
    if reps["Rep_ID"].isna().any():
        warnings.append("Some representatives have missing Rep_ID values.")
    duplicate_reps = int(reps["Rep_ID"].duplicated().sum())
    if duplicate_reps:
        warnings.append(f"{duplicate_reps} duplicate Rep_ID values were found.")
    over_capacity = reps["Current_Capacity"].gt(reps["Max_Capacity"]).fillna(False)
    if over_capacity.any():
        warnings.append(f"{int(over_capacity.sum())} representatives currently exceed their maximum capacity.")
    for label, series in (("leads", leads["Region"]), ("leads", leads["Language"]), ("representatives", reps["Region"]), ("representatives", reps["Language"])):
        missing = int(series.isna().sum())
        if missing:
            warnings.append(f"{missing} {label} have an unknown {series.name.lower().replace('_', ' ')}.")
    return warnings


def match_and_assign(leads, reps, require_region=True, require_language=True):
    assigned = leads.copy()
    updated = reps.copy()
    current = {i: int(row["Current_Capacity"]) for i, row in updated.iterrows()}
    maximum = {i: int(row["Max_Capacity"]) for i, row in updated.iterrows()}
    seniority_rank = {"manager": 3, "senior": 2, "junior": 1}
    rep_ids, rep_names, statuses, scores = [], [], [], []
    for _, lead in assigned.iterrows():
        best_index, best_ranking, best_score = None, None, 0
        for index, rep in updated.iterrows():
            cap, max_cap = current[index], maximum[index]
            if max_cap <= 0 or cap >= max_cap:
                continue
            region_match = regions_overlap(lead["Region"], rep["Region"])
            language_match = pd.notna(lead["Language"]) and pd.notna(rep["Language"]) and lead["Language"] == rep["Language"]
            industry_match = pd.notna(lead["Industry_Vertical"]) and pd.notna(rep["Industry_Vertical_Expertise"]) and lead["Industry_Vertical"] == rep["Industry_Vertical_Expertise"]
            if (require_region and not region_match) or (require_language and not language_match):
                continue
            match_score = 4 * region_match + 3 * language_match + 2 * industry_match
            utilization = cap / max_cap
            seniority = seniority_rank.get(rep["Seniority_Level"], 0)
            ranking = (match_score, seniority, -utilization, max_cap - cap) if lead["Priority"] in {"critical", "high"} else (match_score, -utilization, max_cap - cap, seniority)
            if best_ranking is None or ranking > best_ranking:
                best_index, best_ranking, best_score = index, ranking, match_score
        if best_index is None:
            rep_ids.append(pd.NA); rep_names.append(pd.NA); statuses.append("unassigned"); scores.append(0)
        else:
            selected = updated.loc[best_index]
            rep_ids.append(selected["Rep_ID"]); rep_names.append(selected["Name"]); statuses.append("assigned"); scores.append(best_score)
            current[best_index] += 1
    assigned["Assigned_Rep_ID"] = rep_ids
    assigned["Assigned_Rep_Name"] = rep_names
    assigned["Assignment_Status"] = statuses
    assigned["Match_Score"] = scores
    updated["Final_Current_Capacity"] = [current[i] for i in updated.index]
    updated["Final_Available_Capacity"] = (updated["Max_Capacity"] - updated["Final_Current_Capacity"]).clip(lower=0)
    updated["Final_Capacity_Utilization"] = np.where(updated["Max_Capacity"] > 0, updated["Final_Current_Capacity"] / updated["Max_Capacity"], 1.0)
    updated["New_Leads_Assigned"] = updated["Final_Current_Capacity"] - updated["Current_Capacity"]
    updated = updated.sort_values(["Final_Capacity_Utilization", "Final_Available_Capacity", "Rep_ID"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    return assigned, updated


@st.cache_data(show_spinner="Cleaning and scoring leads...")
def process_leads(file_bytes):
    return score_leads(clean_leads(pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")))


@st.cache_data(show_spinner="Cleaning representatives...")
def process_reps(file_bytes):
    reps = clean_reps(pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig"))
    return reps.sort_values(["Capacity_Utilization", "Available_Capacity", "Rep_ID"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


st.title("Lead Cleaning and Assignment System")
st.write("Upload your leads and sales-representatives CSV files. The app cleans and normalizes data, scores leads, checks capacity, and recommends assignments.")
with st.sidebar:
    st.header("Assignment settings")
    require_region = st.checkbox("Require region match", value=True)
    require_language = st.checkbox("Require language match", value=True)
    st.divider()
    st.markdown("**Lead score rules**\n\n- VIP referral: 100 points\n- Company size: up to 50 points\n- Recent activity: up to 35 points\n- Industry: up to 15 points")

left, right = st.columns(2)
with left:
    lead_file = st.file_uploader("Upload leads CSV", type=["csv"], key="lead_file")
with right:
    rep_file = st.file_uploader("Upload representatives CSV", type=["csv"], key="rep_file")

cleaned_leads = cleaned_reps = None
if lead_file is not None:
    try:
        cleaned_leads = process_leads(lead_file.getvalue())
    except Exception as error:
        st.error(f"Unable to process the leads file: {error}")
if rep_file is not None:
    try:
        cleaned_reps = process_reps(rep_file.getvalue())
    except Exception as error:
        st.error(f"Unable to process the representatives file: {error}")

if cleaned_leads is not None:
    st.divider(); st.header("Lead results")
    metrics = st.columns(5)
    metrics[0].metric("Total leads", f"{len(cleaned_leads):,}")
    for col, priority in zip(metrics[1:4], ("critical", "high", "medium")):
        col.metric(priority.title(), f"{(cleaned_leads['Priority'] == priority).sum():,}")
    metrics[4].metric("Average score", f"{cleaned_leads['Score'].mean() if len(cleaned_leads) else 0:.1f}")
    selected = st.multiselect("Filter leads by priority", ["critical", "high", "medium", "low"], default=["critical", "high", "medium", "low"])
    st.dataframe(cleaned_leads[cleaned_leads["Priority"].isin(selected)], use_container_width=True, hide_index=True)
    st.download_button("Download cleaned leads", dataframe_to_csv(cleaned_leads), "cleaned_leads.csv", "text/csv")

if cleaned_reps is not None:
    st.divider(); st.header("Representative results")
    metrics = st.columns(4)
    metrics[0].metric("Total representatives", f"{len(cleaned_reps):,}")
    metrics[1].metric("Available capacity", f"{cleaned_reps['Available_Capacity'].sum():,.0f}")
    metrics[2].metric("Average utilization", f"{cleaned_reps['Capacity_Utilization'].mean() if len(cleaned_reps) else 0:.1%}")
    metrics[3].metric("At capacity", f"{(cleaned_reps['Available_Capacity'] <= 0).sum():,}")
    display_reps = cleaned_reps.copy()
    display_reps["Capacity_Utilization"] = (display_reps["Capacity_Utilization"] * 100).round(1).astype("string") + "%"
    st.dataframe(display_reps, use_container_width=True, hide_index=True)
    st.download_button("Download cleaned representatives", dataframe_to_csv(cleaned_reps), "cleaned_representatives.csv", "text/csv")

if cleaned_leads is not None and cleaned_reps is not None:
    st.divider(); st.header("Data validation")
    warnings = validate_data(cleaned_leads, cleaned_reps)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("The uploaded data passed all validation checks.")
    st.divider(); st.header("Lead assignment")
    st.write("Leads are assigned from highest to lowest priority based on match quality, seniority (for high-priority leads), and representative capacity.")
    if st.button("Assign leads to representatives", type="primary", use_container_width=True):
        assigned, updated = match_and_assign(cleaned_leads, cleaned_reps, require_region, require_language)
        total_assigned = int((assigned["Assignment_Status"] == "assigned").sum())
        total_unassigned = len(assigned) - total_assigned
        metrics = st.columns(4)
        metrics[0].metric("Total leads", f"{len(assigned):,}")
        metrics[1].metric("Assigned", f"{total_assigned:,}")
        metrics[2].metric("Unassigned", f"{total_unassigned:,}")
        metrics[3].metric("Assignment rate", f"{total_assigned / len(assigned):.1%}" if len(assigned) else "0.0%")
        assignments_tab, capacity_tab = st.tabs(["Lead assignments", "Updated representative capacity"])
        with assignments_tab:
            assignment_filter = st.selectbox("Show assignments", ["All", "Assigned only", "Unassigned only"])
            visible = assigned
            if assignment_filter == "Assigned only":
                visible = assigned[assigned["Assignment_Status"] == "assigned"]
            elif assignment_filter == "Unassigned only":
                visible = assigned[assigned["Assignment_Status"] == "unassigned"]
            st.dataframe(visible, use_container_width=True, hide_index=True)
            st.download_button("Download lead assignments", dataframe_to_csv(assigned), "lead_assignments.csv", "text/csv")
        with capacity_tab:
            display_updated = updated.copy()
            display_updated["Final_Capacity_Utilization"] = (display_updated["Final_Capacity_Utilization"] * 100).round(1).astype("string") + "%"
            st.dataframe(display_updated, use_container_width=True, hide_index=True)
            st.download_button("Download updated representative capacity", dataframe_to_csv(updated), "updated_representative_capacity.csv", "text/csv")

if lead_file is None and rep_file is None:
    st.info("Upload your leads CSV and representatives CSV to begin.")
