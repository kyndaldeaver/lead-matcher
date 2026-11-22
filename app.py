import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Sales Lead Assignment Dashboard")

# --- Tabs layout ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Upload Data", 
    "2. Dashboard", 
    "3. Lead Filtering", 
    "4. Rep Capacity", 
    "5. Lead Assignment Queue",
    "6. Download Cleaned Data"
])

# --- Put content in each tab ---

# 1. Data Upload Tab
with tab1:
    st.header("Upload your CSV files")
    leads_file = st.file_uploader("Upload Leads CSV", type="csv")
    reps_file = st.file_uploader("Upload Reps CSV", type="csv")
    st.info("Please upload both CSV files to proceed. Once uploaded, switch tabs.")

# Only proceed if both files are uploaded
if leads_file and reps_file:
    # --- Data cleaning as before ---
    leads = pd.read_csv(leads_file)
    reps = pd.read_csv(reps_file)

    # Dictionaries for cleaning
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

    # Lead cleaning
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
    # Company size cleaning
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

    # Rep cleaning
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
    labels=["low","medium","high","critical"]
    leads["priority_score"] = pd.cut(leads["score"], bins=bins, labels=labels)
    leads = leads.sort_values("score", ascending=False)

    # Tab 2: Dashboard
    with tab2:
        st.header("Overview Dashboard")
        lead_sample_cols = [c for c in ["Lead_ID", "Company_Name", "Assigned_Rep_ID", "Assigned_Rep_Name"] if c in leads.columns]
        if not lead_sample_cols:
            lead_sample_cols = leads.columns[:2]
        st.dataframe(leads[lead_sample_cols].head(50))
        selected_lead_id = st.selectbox("Select a Lead ID to view details", leads["Lead_ID"])
        lead_info = leads[leads["Lead_ID"] == selected_lead_id]
        st.markdown("**Lead Details:**")
        st.dataframe(lead_info)
        # Pie: priority
        priority_counts = leads["priority_score"].value_counts().sort_index()
        pie2 = go.Figure(go.Pie(labels=priority_counts.index, values=priority_counts.values, textinfo='percent+label'))
        pie2.update_layout(title="Leads by Priority")
        st.plotly_chart(pie2)

        # --- Recent Activity Pie Chart ---
        activity_counts = leads["Recent_activity"].value_counts()
        activity_pie = go.Figure(go.Pie(
            labels=activity_counts.index,
            values=activity_counts.values,
            textinfo='percent+label'
        ))
        activity_pie.update_layout(title="Leads by Recent Activity")
        st.plotly_chart(activity_pie)

    # Tab 3: Lead Filtering
    with tab3:
    st.header("Filter & Sort Leads")
    if "leads" in st.session_state:
        leads = st.session_state.leads

        region_filter = st.multiselect("Region", leads["Region"].unique())
        priority_filter = st.multiselect("Priority", leads["priority_score"].unique())
        language_filter = st.multiselect("Language", leads["Language"].unique())

        # Dynamically generate all possible sort columns
        candidate_sort_cols = list(leads.columns)
        sort_option = st.selectbox("Sort by", candidate_sort_cols)

        sort_direction = st.radio(
            "Sort direction",
            ["Descending", "Ascending"],
            index=0,
            horizontal=True
        )
        ascending = sort_direction == "Ascending"

        filtered = leads.copy()
        if region_filter:
            filtered = filtered[filtered["Region"].isin(region_filter)]
        if priority_filter:
            filtered = filtered[filtered["priority_score"].isin(priority_filter)]
        if language_filter:
            filtered = filtered[filtered["Language"].isin(language_filter)]

        # DOUBLE-CHECK: user can only pick an available column from selectbox, but double-check for safety!
        if sort_option in filtered.columns:
            filtered = filtered.sort_values(sort_option, ascending=ascending)
        else:
            st.warning(f"Sort column '{sort_option}' not present after filtering. Available: {filtered.columns.tolist()}")

        st.dataframe(filtered.head(50))
    else:
        st.info("Upload files in Tab 1 to filter leads.")
        
    # Tab 4: Rep Capacity Editing
    with tab4:
        st.header("Reps with Zero or Negative Max Capacity")
        reps_problem = reps[reps["Max_Capacity"] <= 0]
        st.dataframe(reps_problem)
        if not reps_problem.empty:
            edit_row_id = st.selectbox("Rep to edit", reps_problem["Rep_ID"])
            new_cap = st.number_input("New Max Capacity", min_value=1, max_value=9999, value=100)
            if st.button("Update Rep Capacity"):
                reps.loc[reps["Rep_ID"] == edit_row_id, "Max_Capacity"] = new_cap
                st.success(f"Rep {edit_row_id} capacity updated.")

    # Tab 5: Lead Assignment Queue
    with tab5:
        st.header("Lead Assignment Queue - Unassigned, Highest Priority First")
        queue = leads[leads["Assigned_Rep_ID"].isna()] if "Assigned_Rep_ID" in leads.columns else leads
        queue = queue.sort_values("priority_score", ascending=False)
        st.dataframe(queue.head(50))

    # Tab 6: Download Cleaned Data
    with tab6:
        st.header("Download Cleaned CSVs")
        st.download_button("Download Clean Leads CSV", leads.to_csv(index=False), file_name="leads_cleaned.csv")
        st.download_button("Download Clean Reps CSV", reps.to_csv(index=False), file_name="reps_cleaned.csv")

else:
    with tab2:
        st.info("Upload files in Tab 1 to use dashboard.")
    with tab3:
        st.info("Upload files in Tab 1 to filter leads.")
    with tab4:
        st.info("Upload files in Tab 1 to edit rep capacity.")
    with tab5:
        st.info("Upload files in Tab 1 to see lead assignment queue.")
    with tab6:
        st.info("Upload files in Tab 1 to download cleaned data.")
