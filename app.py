import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load CSVs ---
@st.cache_data
def load_and_clean_data():
    leads = pd.read_csv("leads_scale.csv")
    reps = pd.read_csv("reps_scale.csv")

    # Dictionaries
    region_map = {
        "asa-pacific": "asia pacific", "asia-pacific": "asia pacific", "asia.pacific": "asia pacific",
        "asiapacific": "asia pacific", "apac": "asia pacific",
        "europ": "europe", "europe": "europe", "eur": "europe",
        "asiapacificeurope": "asia pacific and europe",
        "northamerica": "north america", "nortamer": "north america",
        "nortamerica": "north america", "northamer": "north america",
        "northamericaasia-pacific": "north america and asia pacific",
        "northamericaasiapacific": "north america and asia pacific",
    }
    language_map = {
        "es": "spanish", "spanish": "spanish", "en": "english", "english": "english",
        "de": "german", "german": "german",
    }
    activity_map = {
        "n/a": "n/a", "unknown": "unknown", "subscribed": "subscribed", "": "blank",
        "downloadedwhitepaper": "Downloaded Whitepaper",
        "requestedlivedemo": "Requested Live Demo",
        "openedmarketingemail": "Opened Market Email",
        "contactsales": "Contact Sales",
        "visitedpricingpage": "Visited Pricing Page",
        "startedfreetrial": "Started Free Trial",
        "attendedwebinar": "Attended Webinar",
        "subscribedtoblog": "Subscribed To Blog"
    }
    vip_map = {"true": "true", "false": "false", "unknown": "unknown"}
    seniority_map = {"mgr": "manager", "sr": "senior", "junior": "junior", "jr": "junior", "senior": "senior", "manager": "manager"}

    # Vectorized cleaning - leads
    leads["Region"] = (
        leads["Region"].astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
        .map(region_map).fillna("N/A")
    )
    leads["Language"] = leads["Language"].astype("string").str.strip().str.lower().map(language_map).fillna("N/A")
    leads["Industry_Vertical"] = leads["Industry_Vertical"].astype("string").str.strip().str.lower()
    leads["VIP_Referral"] = leads["VIP_Referral"].astype("string").str.strip().str.lower().map(vip_map).fillna("N/A")
    leads["Recent_activity"] = (
        leads["Recent_Activity"].astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
        .map(activity_map).fillna("N/A")
    )
    # Company size vectorized
    def clean_company_size(s):
        s = s.astype(str).str.strip().str.lower()
        def _clean(val):
            if val in ["none", "nan", "n/a", ""]:
                return "N/A"
            if "k" in val:
                return int(float(val.replace("k", "")) * 1000)
            try:
                return int(float(val))
            except: return "N/A"
        return s.apply(_clean)
    leads["Company_Size"] = clean_company_size(leads["Company_Size"])

    # Reps cleaning
    reps["Name"] = reps["Name"].astype(str).str.split("(").str[0].str.strip()
    reps["Region"] = (
        reps["Region"].astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
        .map(region_map).fillna("N/A")
    )
    reps["Language"] = reps["Language"].astype("string").str.strip().str.lower().map(language_map).fillna("N/A")
    reps["Industry_Vertical_Expertise"] = reps["Industry_Vertical_Expertise"].astype("string").str.strip().str.lower()
    reps["Seniority_Level"] = reps["Seniority_Level"].astype("string").str.strip().str.lower().map(seniority_map).fillna("N/A")

    reps["Max_Capacity"] = pd.to_numeric(reps["Max_Capacity"], errors="coerce").fillna(0)
    reps["Current_Capacity"] = pd.to_numeric(reps["Current_Capacity"], errors="coerce").fillna(0)
    reps["percentage_capacity"] = reps.apply(
        lambda row: 1 if row["Max_Capacity"] <= 0 else (row["Current_Capacity"]/row["Max_Capacity"]), axis=1
    )
    reps = reps.sort_values("percentage_capacity", ascending=True)

    # Lead scoring
    def score_lead(row):
        if row["VIP_Referral"] == "true":
            return 100
        score = 0
        size = row["Company_Size"]
        if isinstance(size, (int,float)) and size >= 1000:
            score += 50
        elif isinstance(size, (int,float)) and size >= 100:
            score += 30
        elif size == "N/A":
            score += 20
        # Recent activity
        act = row["Recent_activity"]
        if act in ["Requested Live Demo", "Contact Sales"]:
            score += 35
        elif act in ["Attended Webinar", "Started Free Trial"]:
            score += 25
        elif act == "Downloaded Whitepaper":
            score += 15
        elif act in ["Subscribed To Blog", "subscribed"]:
            score += 5
        elif act == "Visited Pricing Page":
            score += 10
        # Industry scoring
        if row["Industry_Vertical"] == "technology":
            score += 15
        elif row["Industry_Vertical"] in ["healthcare", "finance"]:
            score += 10
        return score

    leads["score"] = leads.apply(score_lead, axis=1)
    # Priority labels
    bins = [-float('inf'),40,60,85,float('inf')]
    labels=["low","medium","high","critical"]
    leads["priority_score"] = pd.cut(leads["score"], bins=bins, labels=labels)
    # Sorted lead list for assignment queue
    leads = leads.sort_values("score", ascending=False)
    return leads, reps

leads, reps = load_and_clean_data()

st.title("Sales Lead Assignment Dashboard")

# --- Table with clickable details ---
st.subheader("Leads Table")
main_cols = ["Lead_ID", "Company_Name", "Assigned_Rep_ID", "Assigned_Rep_Name"] if "Assigned_Rep_ID" in leads else leads.columns[:2]
st.dataframe(leads[main_cols].head(50), use_container_width=True)

selected_lead_id = st.selectbox("Select a Lead ID to view details", leads["Lead_ID"])
lead_info = leads[leads["Lead_ID"]==selected_lead_id]
st.markdown("**Lead Details:**")
st.dataframe(lead_info, use_container_width=True)

# --- Pie charts ---

st.subheader("Assignment Status Pie Charts")

assigned = leads["Assigned_Rep_ID"].notna().sum() if "Assigned_Rep_ID" in leads else 0
unassigned = leads["Assigned_Rep_ID"].isna().sum() if "Assigned_Rep_ID" in leads else leads.shape[0]
pie1 = go.Figure(
    go.Pie(
        labels=['Assigned', 'Unassigned'],
        values=[assigned, unassigned],
        textinfo='percent+label'
    )
)
pie1.update_layout(title="Lead Assignment Breakdown")
st.plotly_chart(pie1, use_container_width=True)

priority_counts = leads["priority_score"].value_counts().sort_index()
pie2 = go.Figure(
    go.Pie(
        labels=priority_counts.index,
        values=priority_counts.values,
        textinfo='percent+label'
    )
)
pie2.update_layout(title="Leads by Priority")
st.plotly_chart(pie2, use_container_width=True)

# --- Filters and sorting ---
st.subheader("Filter & Sort Leads")
region_filter = st.multiselect("Region", leads["Region"].unique())
priority_filter = st.multiselect("Priority", leads["priority_score"].unique())
sort_option = st.selectbox("Sort by", ["score", "Company_Size", "Lead_ID"])

filtered = leads.copy()
if region_filter:
    filtered = filtered[filtered["Region"].isin(region_filter)]
if priority_filter:
    filtered = filtered[filtered["priority_score"].isin(priority_filter)]
filtered = filtered.sort_values(sort_option, ascending=False)
st.dataframe(filtered.head(50), use_container_width=True)

# --- Rep capacity tab ---
st.subheader("Reps with Zero or Negative Max Capacity")
reps_problem = reps[reps["Max_Capacity"] <= 0]
st.dataframe(reps_problem, use_container_width=True)
if not reps_problem.empty:
    edit_row = st.selectbox("Rep to edit", reps_problem["Rep_ID"])
    new_cap = st.number_input("New Max Capacity", min_value=1, max_value=9999, value=100)
    if st.button("Update Rep Capacity"):
        reps.loc[reps["Rep_ID"]==edit_row, "Max_Capacity"] = new_cap
        st.success(f"Rep {edit_row} capacity updated.")

# --- Lead Assignment Queue ---
st.subheader("Lead Assignment Queue - Unassigned, Highest Priority First")
queue = leads[leads["Assigned_Rep_ID"].isna()] if "Assigned_Rep_ID" in leads else leads
queue = queue.sort_values("priority_score", ascending=False)
st.dataframe(queue.head(50), use_container_width=True)

# --- Download buttons ---
st.download_button("Download Clean Leads CSV", leads.to_csv(index=False), file_name="leads_cleaned.csv")
st.download_button("Download Clean Reps CSV", reps.to_csv(index=False), file_name="reps_cleaned.csv")
