import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Sales Lead Assignment Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Upload Data",
    "2. Dashboard",
    "3. Lead Filtering",
    "4. Rep Capacity",
    "5. Lead Assignment Queue",
    "6. Download Cleaned Data"
])

# --- Tab 1: Upload ---
with tab1:
    st.header("Upload your CSV files")
    leads_file = st.file_uploader("Upload Leads CSV", type="csv")
    reps_file = st.file_uploader("Upload Reps CSV", type="csv")
    st.info("Please upload both CSV files to proceed. Once uploaded, switch tabs.")

# --- CLEANING LOGIC ---
def clean_data(leads, reps):
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
        "downloadedwhitepaper": "Downloaded Whitepaper", "requestedlivedemo": "Requested Live Demo",
        "openedmarketingemail": "Opened Market Email", "contactsales": "Contact Sales",
        "visitedpricingpage": "Visited Pricing Page", "startedfreetrial": "Started Free Trial",
        "attendedwebinar": "Attended Webinar", "subscribedtoblog": "Subscribed To Blog"
    }
    vip_map = {"true": "true", "false": "false", "unknown": "unknown"}
    seniority_map = {"mgr": "manager", "sr": "senior", "junior": "junior", "jr": "junior", "senior": "senior", "manager": "manager"}

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
    # Clean company size
    def clean_company_size(s):
        def _clean(val):
            if pd.isna(val) or str(val).strip().lower() in ["none", "nan", "n/a", ""]:
                return "N/A"
            val_str = str(val).strip().lower()
            if "k" in val_str:
                try:
                    return int(float(val_str.replace("k", "")) * 1000)
                except:
                    return "N/A"
            try:
                return int(float(val_str))
            except:
                return "N/A"
        return s.apply(_clean)
    leads["Company_Size"] = clean_company_size(leads["Company_Size"])

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
    bins = [-float('inf'),40,60,85,float('inf')]
    labels = ["low","medium","high","critical"]
    leads["priority_score"] = pd.cut(leads["score"], bins=bins, labels=labels)
    leads = leads.sort_values("score", ascending=False)

    # Ensure Assigned columns exist, and clean up any non-NA empty/unassigned values
    if "Assigned_Rep_ID" not in leads.columns:
        leads["Assigned_Rep_ID"] = pd.NA
    else:
        leads["Assigned_Rep_ID"] = leads["Assigned_Rep_ID"].replace(["", "N/A", "None"], pd.NA)
    if "Assigned_Rep_Name" not in leads.columns:
        leads["Assigned_Rep_Name"] = pd.NA
    else:
        leads["Assigned_Rep_Name"] = leads["Assigned_Rep_Name"].replace(["", "N/A", "None"], pd.NA)

    return leads, reps

# ---- SESSION STATE HANDLE ----
if "leads" not in st.session_state and "reps" not in st.session_state and leads_file and reps_file:
    leads = pd.read_csv(leads_file)
    reps = pd.read_csv(reps_file)
    leads, reps = clean_data(leads, reps)
    st.session_state.leads = leads
    st.session_state.reps = reps

# ---- Dashboard Tab ----
with tab2:
    st.header("Overview Dashboard")
    if ("leads" in st.session_state) and ("reps" in st.session_state):
        leads = st.session_state.leads
        reps = st.session_state.reps

        lead_sample_cols = [c for c in ["Lead_ID", "Company_Name", "Assigned_Rep_ID", "Assigned_Rep_Name"] if c in leads.columns]
        if not lead_sample_cols:
            lead_sample_cols = leads.columns[:2]
        st.dataframe(leads[lead_sample_cols].head(50))

        selected_lead_id = st.selectbox("Select a Lead ID to view details", leads["Lead_ID"])
        lead_info = leads[leads["Lead_ID"] == selected_lead_id]
        st.markdown("**Lead Details:**")
        st.dataframe(lead_info)

        # Diagnostic section (NEW!)
        st.markdown("### Diagnostic: Assigned_Rep_ID value counts (including NA):")
        st.write(leads["Assigned_Rep_ID"].value_counts(dropna=False))
        unassigned_count = leads["Assigned_Rep_ID"].isna().sum()
        assigned_count = leads["Assigned_Rep_ID"].notna().sum()
        st.write(f"Assigned: {assigned_count}, Unassigned: {unassigned_count}")

        # Assignment button
        st.markdown("### Assign Leads to Reps")
        if st.button("Assign All Unassigned Leads"):
            reps_capacity = reps.copy()
            reps_capacity["leads_assigned"] = 0
            for idx, lead in leads.iterrows():
                if pd.notna(lead["Assigned_Rep_ID"]):
                    continue  # Already assigned

                # Only consider reps with remaining capacity in region
                possible_reps = reps_capacity[
                    (reps_capacity["Region"] == lead["Region"]) &
                    (reps_capacity["leads_assigned"] < reps_capacity["Max_Capacity"])
                ]
                if possible_reps.empty:
                    # If nobody in region, try any rep with capacity
                    possible_reps = reps_capacity[reps_capacity["leads_assigned"] < reps_capacity["Max_Capacity"]]
                if possible_reps.empty:
                    continue  # No rep left with capacity

                # Pick rep with lowest percentage capacity
                rep_row = possible_reps.sort_values("percentage_capacity").iloc[0]
                leads.at[idx, "Assigned_Rep_ID"] = rep_row["Rep_ID"]
                leads.at[idx, "Assigned_Rep_Name"] = rep_row["Name"]
                reps_capacity.loc[rep_row.name, "leads_assigned"] += 1
            st.session_state.leads = leads
            st.success("Leads assigned! Refresh dashboard/queue tabs to see updates.")

        # Pie chart
        pie1 = go.Figure(go.Pie(
            labels=['Assigned', 'Unassigned'],
            values=[assigned_count, unassigned_count],
            textinfo='percent+label'))
        pie1.update_layout(title="Lead Assignment Breakdown")
        st.plotly_chart(pie1)
        priority_counts = leads["priority_score"].value_counts().sort_index()
        pie2 = go.Figure(go.Pie(labels=priority_counts.index, values=priority_counts.values, textinfo='percent+label'))
        pie2.update_layout(title="Leads by Priority")
        st.plotly_chart(pie2)
    else:
        st.info("Upload files in Tab 1 to use dashboard.")

# ---- Filtering Tab ----
with tab3:
    st.header("Filter & Sort Leads")
    if ("leads" in st.session_state):
        leads = st.session_state.leads
        region_filter = st.multiselect("Region", leads["Region"].unique())
        priority_filter = st.multiselect("Priority", leads["priority_score"].unique())
        sort_option = st.selectbox("Sort by", ["score", "Company_Size", "Lead_ID"])
        filtered = leads.copy()
        if region_filter:
            filtered = filtered[filtered["Region"].isin(region_filter)]
        if priority_filter:
            filtered = filtered[filtered["priority_score"].isin(priority_filter)]
        filtered = filtered.sort_values(sort_option, ascending=False)
        st.dataframe(filtered.head(50))
    else:
        st.info("Upload files in Tab 1 to filter leads.")

# ---- Rep Capacity Tab ----
with tab4:
    st.header("Reps with Zero or Negative Max Capacity")
    if ("reps" in st.session_state):
        reps = st.session_state.reps
        reps_problem = reps[reps["Max_Capacity"] <= 0]
        st.dataframe(reps_problem)
        if not reps_problem.empty:
            edit_row_id = st.selectbox("Rep to edit", reps_problem["Rep_ID"])
            new_cap = st.number_input("New Max Capacity", min_value=1, max_value=9999, value=100)
            if st.button("Update Rep Capacity"):
                reps.loc[reps["Rep_ID"] == edit_row_id, "Max_Capacity"] = new_cap
                st.session_state.reps = reps
                st.success(f"Rep {edit_row_id} capacity updated.")
    else:
        st.info("Upload files in Tab 1 to edit rep capacity.")

# ---- Assignment Queue Tab ----
with tab5:
    st.header("Lead Assignment Queue - Unassigned, Highest Priority First")
    if ("leads" in st.session_state):
        leads = st.session_state.leads
        queue = leads[leads["Assigned_Rep_ID"].isna()]
        queue = queue.sort_values("priority_score", ascending=False)
        st.dataframe(queue.head(50))
    else:
        st.info("Upload files in Tab 1 to see lead assignment queue.")

# ---- Download Data Tab ----
with tab6:
    st.header("Download Cleaned CSVs")
    if ("leads" in st.session_state):
        leads = st.session_state.leads
        st.download_button("Download Clean Leads CSV", leads.to_csv(index=False), file_name="leads_cleaned.csv")
    else:
        st.info("Upload files in Tab 1 to download leads data.")
    if ("reps" in st.session_state):
        reps = st.session_state.reps
        st.download_button("Download Clean Reps CSV", reps.to_csv(index=False), file_name="reps_cleaned.csv")
    else:
        st.info("Upload files in Tab 1 to download reps data.")
