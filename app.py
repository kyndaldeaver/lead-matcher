import io

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Team 36's Lead Assignment System",
    page_icon=" ",
    layout="wide",
)


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

LEAD_COLUMNS = {
    "Lead_ID",
    "Company_Name",
    "Region",
    "Language",
    "Industry_Vertical",
    "Company_Size",
    "Recent_Activity",
    "VIP_Referral",
}

REP_COLUMNS = {
    "Rep_ID",
    "Name",
    "Region",
    "Language",
    "Industry_Vertical_Expertise",
    "Current_Capacity",
    "Max_Capacity",
}


# ============================================================
# NORMALIZATION DICTIONARIES
# ============================================================

REGION_MAP = {
    "asapacific": "asia pacific",
    "asiapacific": "asia pacific",
    "apac": "asia pacific",

    "europ": "europe",
    "europe": "europe",
    "eur": "europe",

    "northamerica": "north america",
    "nortamer": "north america",
    "nortamerica": "north america",
    "northamer": "north america",

    "asiapacificeurope": "asia pacific and europe",
    "northamericaasiapacific": "north america and asia pacific",
}

LANGUAGE_MAP = {
    "en": "english",
    "english": "english",

    "es": "spanish",
    "spanish": "spanish",

    "de": "german",
    "german": "german",
}

ACTIVITY_MAP = {
    "subscribed": "subscribed",
    "subscribedtoblog": "subscribed to blog",
    "downloadedwhitepaper": "downloaded whitepaper",
    "requestedlivedemo": "requested live demo",
    "openedmarketingemail": "opened marketing email",
    "contactsales": "contact sales",
    "visitedpricingpage": "visited pricing page",
    "startedfreetrial": "started free trial",
    "attendedwebinar": "attended webinar",
    "unknown": "unknown",
    "na": "unknown",
}

SENIORITY_MAP = {
    "mgr": "manager",
    "manager": "manager",

    "sr": "senior",
    "senior": "senior",

    "jr": "junior",
    "junior": "junior",
}


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def require_columns(dataframe, required_columns, file_name):
    """Verify that an uploaded file contains all required columns."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"{file_name} is missing the following columns: "
            f"{missing_text}"
        )


def normalize_text(series):
    """Trim whitespace, convert to lowercase, and handle missing values."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
            "n/a": pd.NA,
            "na": pd.NA,
        })
    )


def normalize_key(series):
    """
    Create dictionary lookup keys by removing spaces and punctuation.

    Example:
        "Asia-Pacific" -> "asiapacific"
    """

    return (
        normalize_text(series)
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


def map_normalized(series, mapping):
    """Normalize a column and apply a standard-value dictionary."""

    keys = normalize_key(series)
    return keys.map(mapping).astype("string")


def dataframe_to_csv(dataframe):
    """Convert a DataFrame into downloadable CSV bytes."""

    return dataframe.to_csv(index=False).encode("utf-8")


def clean_company_size(series):
    """
    Convert company-size values to numbers.

    Examples:
        500     -> 500
        2.5k    -> 2500
        1,200   -> 1200
    """

    text = (
        normalize_text(series)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    number = pd.to_numeric(
        text.str.extract(
            r"([0-9]*\.?[0-9]+)",
            expand=False,
        ),
        errors="coerce",
    )

    contains_k = text.str.endswith("k", na=False)
    multiplier = np.where(contains_k, 1000, 1)

    result = number * multiplier

    return result.round().astype("Int64")


def regions_overlap(region_one, region_two):
    """
    Determine whether two normalized regions overlap.

    For example, "asia pacific and europe" overlaps with "europe".
    """

    if pd.isna(region_one) or pd.isna(region_two):
        return False

    regions_one = {
        item.strip()
        for item in str(region_one).split(" and ")
        if item.strip()
    }

    regions_two = {
        item.strip()
        for item in str(region_two).split(" and ")
        if item.strip()
    }

    return bool(regions_one.intersection(regions_two))


# ============================================================
# LEAD CLEANING
# ============================================================

def clean_leads(leads):
    """Clean and normalize an uploaded leads DataFrame."""

    require_columns(
        leads,
        LEAD_COLUMNS,
        "The leads file",
    )

    leads = leads.copy()

    leads["Lead_ID"] = (
        leads["Lead_ID"]
        .astype("string")
        .str.strip()
    )

    leads["Company_Name"] = (
        leads["Company_Name"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    leads["Region"] = map_normalized(
        leads["Region"],
        REGION_MAP,
    )

    leads["Language"] = map_normalized(
        leads["Language"],
        LANGUAGE_MAP,
    )

    leads["Industry_Vertical"] = normalize_text(
        leads["Industry_Vertical"]
    )

    leads["Company_Size"] = clean_company_size(
        leads["Company_Size"]
    )

    leads["Recent_Activity"] = (
        map_normalized(
            leads["Recent_Activity"],
            ACTIVITY_MAP,
        )
        .fillna("unknown")
    )

    vip_text = normalize_text(leads["VIP_Referral"])

    leads["VIP_Referral"] = (
        vip_text.map({
            "true": True,
            "yes": True,
            "1": True,

            "false": False,
            "no": False,
            "0": False,

            "unknown": pd.NA,
        })
        .astype("boolean")
    )

    return leads


# ============================================================
# REPRESENTATIVE CLEANING
# ============================================================

def clean_reps(reps):
    """Clean and normalize an uploaded representatives DataFrame."""

    require_columns(
        reps,
        REP_COLUMNS,
        "The representatives file",
    )

    reps = reps.copy()

    reps["Rep_ID"] = (
        reps["Rep_ID"]
        .astype("string")
        .str.strip()
    )

    # Example: "Maria Fernandez (West)" -> "Maria Fernandez"
    reps["Name"] = (
        reps["Name"]
        .astype("string")
        .str.replace(
            r"\s*\([^)]*\)\s*",
            " ",
            regex=True,
        )
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    reps["Region"] = map_normalized(
        reps["Region"],
        REGION_MAP,
    )

    reps["Language"] = map_normalized(
        reps["Language"],
        LANGUAGE_MAP,
    )

    reps["Industry_Vertical_Expertise"] = normalize_text(
        reps["Industry_Vertical_Expertise"]
    )

    reps["Seniority_Level"] = map_normalized(
        reps["Seniority_Level"],
        SENIORITY_MAP,
    )

    reps["Current_Capacity"] = (
        pd.to_numeric(
            reps["Current_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype("Int64")
    )

    reps["Max_Capacity"] = (
        pd.to_numeric(
            reps["Max_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype("Int64")
    )

    current_capacity = reps["Current_Capacity"].astype("Float64")
    max_capacity = reps["Max_Capacity"].astype("Float64")

    reps["Capacity_Utilization"] = (
        current_capacity
        .div(max_capacity.where(max_capacity > 0))
        .fillna(1.0)
        .clip(lower=0)
    )

    reps["Available_Capacity"] = (
        reps["Max_Capacity"] - reps["Current_Capacity"]
    ).clip(lower=0)

    return reps


# ============================================================
# LEAD SCORING
# ============================================================

def score_leads(leads):
    """Calculate lead scores and priority categories."""

    leads = leads.copy()

    score = pd.Series(
        0,
        index=leads.index,
        dtype="int64",
    )

    # Company-size scoring
    company_size_score = np.select(
        [
            leads["Company_Size"].ge(1000).fillna(False),
            leads["Company_Size"].ge(100).fillna(False),
            leads["Company_Size"].isna(),
        ],
        [
            50,
            30,
            20,
        ],
        default=0,
    )

    score += company_size_score

    # Recent-activity scoring
    activity_points = {
        "requested live demo": 35,
        "contact sales": 35,
        "attended webinar": 25,
        "started free trial": 25,
        "downloaded whitepaper": 15,
        "visited pricing page": 10,
        "subscribed to blog": 5,
        "subscribed": 5,
    }

    score += (
        leads["Recent_Activity"]
        .map(activity_points)
        .fillna(0)
        .astype("int64")
    )

    # Industry scoring
    industry_points = {
        "technology": 15,
        "healthcare": 10,
        "finance": 10,
    }

    score += (
        leads["Industry_Vertical"]
        .map(industry_points)
        .fillna(0)
        .astype("int64")
    )

    # VIP leads receive a score of 100.
    is_vip = (
        leads["VIP_Referral"]
        .fillna(False)
        .astype(bool)
    )

    leads["Score"] = score.mask(is_vip, 100)

    leads["Priority"] = np.select(
        [
            leads["Score"].ge(85),
            leads["Score"].ge(60),
            leads["Score"].ge(40),
        ],
        [
            "critical",
            "high",
            "medium",
        ],
        default="low",
    )

    return (
        leads.sort_values(
            by=["Score", "Lead_ID"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_data(leads, reps):
    """Return a list of data-quality warnings."""

    warnings = []

    if leads["Lead_ID"].isna().any():
        warnings.append("Some leads have missing Lead_ID values.")

    duplicate_leads = leads["Lead_ID"].duplicated().sum()

    if duplicate_leads:
        warnings.append(
            f"{duplicate_leads} duplicate Lead_ID values were found."
        )

    if reps["Rep_ID"].isna().any():
        warnings.append("Some representatives have missing Rep_ID values.")

    duplicate_reps = reps["Rep_ID"].duplicated().sum()

    if duplicate_reps:
        warnings.append(
            f"{duplicate_reps} duplicate Rep_ID values were found."
        )

    over_capacity = (
        reps["Current_Capacity"]
        .gt(reps["Max_Capacity"])
        .fillna(False)
    )

    if over_capacity.any():
        warnings.append(
            f"{over_capacity.sum()} representatives currently exceed "
            "their maximum capacity."
        )

    missing_lead_regions = leads["Region"].isna().sum()

    if missing_lead_regions:
        warnings.append(
            f"{missing_lead_regions} leads have an unknown region."
        )

    missing_lead_languages = leads["Language"].isna().sum()

    if missing_lead_languages:
        warnings.append(
            f"{missing_lead_languages} leads have an unknown language."
        )

    missing_rep_regions = reps["Region"].isna().sum()

    if missing_rep_regions:
        warnings.append(
            f"{missing_rep_regions} representatives have an unknown region."
        )

    missing_rep_languages = reps["Language"].isna().sum()

    if missing_rep_languages:
        warnings.append(
            f"{missing_rep_languages} representatives have an unknown language."
        )

    return warnings


# ============================================================
# LEAD-TO-REPRESENTATIVE ASSIGNMENT
# ============================================================

def match_and_assign(
    leads,
    reps,
    require_region=True,
    require_language=True,
):
    """
    Assign each lead to an available representative.

    Representatives are ranked using:
        1. Region match
        2. Language match
        3. Industry expertise match
        4. Lowest capacity utilization
        5. Seniority for high-priority leads
    """

    assigned_leads = leads.copy()
    updated_reps = reps.copy()

    # Store changing capacity values separately.
    current_capacity = {
        index: int(row["Current_Capacity"])
        for index, row in updated_reps.iterrows()
    }

    max_capacity = {
        index: int(row["Max_Capacity"])
        for index, row in updated_reps.iterrows()
    }

    assigned_rep_ids = []
    assigned_rep_names = []
    assignment_statuses = []
    match_scores = []

    seniority_ranking = {
        "manager": 3,
        "senior": 2,
        "junior": 1,
    }

    for _, lead in assigned_leads.iterrows():
        best_rep_index = None
        best_ranking = None
        best_match_score = 0

        for rep_index, rep in updated_reps.iterrows():
            current = current_capacity[rep_index]
            maximum = max_capacity[rep_index]

            # Do not assign to representatives with no available space.
            if maximum <= 0 or current >= maximum:
                continue

            region_match = regions_overlap(
                lead["Region"],
                rep["Region"],
            )

            language_match = (
                pd.notna(lead["Language"])
                and pd.notna(rep["Language"])
                and lead["Language"] == rep["Language"]
            )

            industry_match = (
                pd.notna(lead["Industry_Vertical"])
                and pd.notna(rep["Industry_Vertical_Expertise"])
                and lead["Industry_Vertical"]
                == rep["Industry_Vertical_Expertise"]
            )

            if require_region and not region_match:
                continue

            if require_language and not language_match:
                continue

            match_score = 0

            if region_match:
                match_score += 4

            if language_match:
                match_score += 3

            if industry_match:
                match_score += 2

            utilization = current / maximum

            seniority_score = seniority_ranking.get(
                rep["Seniority_Level"],
                0,
            )

            # Higher match score is better.
            # Lower utilization is better.
            # More senior representatives are preferred for
            # critical and high-priority leads.
            if lead["Priority"] in {"critical", "high"}:
                ranking = (
                    match_score,
                    seniority_score,
                    -utilization,
                    maximum - current,
                )
            else:
                ranking = (
                    match_score,
                    -utilization,
                    maximum - current,
                    seniority_score,
                )

            if best_ranking is None or ranking > best_ranking:
                best_ranking = ranking
                best_rep_index = rep_index
                best_match_score = match_score

        if best_rep_index is None:
            assigned_rep_ids.append(pd.NA)
            assigned_rep_names.append(pd.NA)
            assignment_statuses.append("unassigned")
            match_scores.append(0)
            continue

        selected_rep = updated_reps.loc[best_rep_index]

        assigned_rep_ids.append(selected_rep["Rep_ID"])
        assigned_rep_names.append(selected_rep["Name"])
        assignment_statuses.append("assigned")
        match_scores.append(best_match_score)

        current_capacity[best_rep_index] += 1

    assigned_leads["Assigned_Rep_ID"] = assigned_rep_ids
    assigned_leads["Assigned_Rep_Name"] = assigned_rep_names
    assigned_leads["Assignment_Status"] = assignment_statuses
    assigned_leads["Match_Score"] = match_scores

    updated_reps["Final_Current_Capacity"] = [
        current_capacity[index]
        for index in updated_reps.index
    ]

    updated_reps["Final_Available_Capacity"] = (
        updated_reps["Max_Capacity"]
        - updated_reps["Final_Current_Capacity"]
    ).clip(lower=0)

    updated_reps["Final_Capacity_Utilization"] = np.where(
        updated_reps["Max_Capacity"] > 0,
        (
            updated_reps["Final_Current_Capacity"]
            / updated_reps["Max_Capacity"]
        ),
        1.0,
    )

    updated_reps["New_Leads_Assigned"] = (
        updated_reps["Final_Current_Capacity"]
        - updated_reps["Current_Capacity"]
    )

    updated_reps = (
        updated_reps.sort_values(
            by=[
                "Final_Capacity_Utilization",
                "Final_Available_Capacity",
                "Rep_ID",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return assigned_leads, updated_reps


# ============================================================
# CACHED FILE PROCESSING
# ============================================================

@st.cache_data(show_spinner="Cleaning and scoring leads...")
def process_leads(file_bytes):
    """Read, clean, and score a leads CSV."""

    leads = pd.read_csv(io.BytesIO(file_bytes))
    leads = clean_leads(leads)
    leads = score_leads(leads)

    return leads


@st.cache_data(show_spinner="Cleaning representatives...")
def process_reps(file_bytes):
    """Read and clean a representatives CSV."""

    reps = pd.read_csv(io.BytesIO(file_bytes))
    reps = clean_reps(reps)

    reps = (
        reps.sort_values(
            by=[
                "Capacity_Utilization",
                "Available_Capacity",
                "Rep_ID",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return reps


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("📊 Lead Cleaning and Assignment System")

st.write(
    """
    Upload your leads and sales-representatives CSV files. The application
    will clean and normalize the data, calculate lead priority scores,
    evaluate representative capacity, and assign leads to appropriate
    representatives.
    """
)

with st.sidebar:
    st.header("Assignment settings")

    require_region = st.checkbox(
        "Require region match",
        value=True,
        help="A lead will only be assigned to a representative "
             "who supports the lead's region.",
    )

    require_language = st.checkbox(
        "Require language match",
        value=True,
        help="A lead will only be assigned to a representative "
             "who speaks the lead's language.",
    )

    st.divider()

    st.markdown(
        """
        **Lead score rules**

        - VIP referral: 100 points
        - Company size: up to 50 points
        - Recent activity: up to 35 points
        - Industry: up to 15 points
        """
    )


upload_column_one, upload_column_two = st.columns(2)

with upload_column_one:
    lead_file = st.file_uploader(
        "Upload leads CSV",
        type=["csv"],
        key="lead_file",
    )

with upload_column_two:
    rep_file = st.file_uploader(
        "Upload representatives CSV",
        type=["csv"],
        key="rep_file",
    )


cleaned_leads = None
cleaned_reps = None


# ============================================================
# PROCESS LEADS
# ============================================================

if lead_file is not None:
    try:
        cleaned_leads = process_leads(
            lead_file.getvalue()
        )

    except Exception as error:
        st.error(
            f"Unable to process the leads file: {error}"
        )


# ============================================================
# PROCESS REPRESENTATIVES
# ============================================================

if rep_file is not None:
    try:
        cleaned_reps = process_reps(
            rep_file.getvalue()
        )

    except Exception as error:
        st.error(
            f"Unable to process the representatives file: {error}"
        )


# ============================================================
# DISPLAY CLEANED LEADS
# ============================================================

if cleaned_leads is not None:
    st.divider()
    st.header("Lead results")

    lead_metric_columns = st.columns(5)

    lead_metric_columns[0].metric(
        "Total leads",
        f"{len(cleaned_leads):,}",
    )

    lead_metric_columns[1].metric(
        "Critical",
        f"{(cleaned_leads['Priority'] == 'critical').sum():,}",
    )

    lead_metric_columns[2].metric(
        "High",
        f"{(cleaned_leads['Priority'] == 'high').sum():,}",
    )

    lead_metric_columns[3].metric(
        "Medium",
        f"{(cleaned_leads['Priority'] == 'medium').sum():,}",
    )

    average_score = (
        cleaned_leads["Score"].mean()
        if not cleaned_leads.empty
        else 0
    )

    lead_metric_columns[4].metric(
        "Average score",
        f"{average_score:.1f}",
    )

    selected_priorities = st.multiselect(
        "Filter leads by priority",
        options=[
            "critical",
            "high",
            "medium",
            "low",
        ],
        default=[
            "critical",
            "high",
            "medium",
            "low",
        ],
    )

    displayed_leads = cleaned_leads[
        cleaned_leads["Priority"].isin(selected_priorities)
    ]

    st.dataframe(
        displayed_leads,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download cleaned leads",
        data=dataframe_to_csv(cleaned_leads),
        file_name="cleaned_leads.csv",
        mime="text/csv",
    )


# ============================================================
# DISPLAY CLEANED REPRESENTATIVES
# ============================================================

if cleaned_reps is not None:
    st.divider()
    st.header("Representative results")

    rep_metric_columns = st.columns(4)

    rep_metric_columns[0].metric(
        "Total representatives",
        f"{len(cleaned_reps):,}",
    )

    rep_metric_columns[1].metric(
        "Available capacity",
        f"{cleaned_reps['Available_Capacity'].sum():,.0f}",
    )

    average_utilization = (
        cleaned_reps["Capacity_Utilization"].mean()
        if not cleaned_reps.empty
        else 0
    )

    rep_metric_columns[2].metric(
        "Average utilization",
        f"{average_utilization:.1%}",
    )

    full_reps = (
        cleaned_reps["Available_Capacity"] <= 0
    ).sum()

    rep_metric_columns[3].metric(
        "Representatives at capacity",
        f"{full_reps:,}",
    )

    displayed_reps = cleaned_reps.copy()

    displayed_reps["Capacity_Utilization"] = (
        displayed_reps["Capacity_Utilization"]
        .mul(100)
        .round(1)
        .astype("string")
        + "%"
    )

    st.dataframe(
        displayed_reps,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download cleaned representatives",
        data=dataframe_to_csv(cleaned_reps),
        file_name="cleaned_representatives.csv",
        mime="text/csv",
    )


# ============================================================
# VALIDATION AND ASSIGNMENT
# ============================================================

if cleaned_leads is not None and cleaned_reps is not None:
    st.divider()
    st.header("Data validation")

    validation_warnings = validate_data(
        cleaned_leads,
        cleaned_reps,
    )

    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    else:
        st.success("The uploaded data passed all validation checks.")

    st.divider()
    st.header("Lead assignment")

    st.write(
        """
        Leads are processed from highest to lowest priority. Each lead is
        assigned to the best available representative according to region,
        language, industry expertise, seniority, and current utilization.
        """
    )

    if st.button(
        "Assign leads to representatives",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Assigning leads..."):
            assigned_leads, updated_reps = match_and_assign(
                cleaned_leads,
                cleaned_reps,
                require_region=require_region,
                require_language=require_language,
            )

        total_assigned = (
            assigned_leads["Assignment_Status"] == "assigned"
        ).sum()

        total_unassigned = (
            assigned_leads["Assignment_Status"] == "unassigned"
        ).sum()

        assignment_rate = (
            total_assigned / len(assigned_leads)
            if len(assigned_leads) > 0
            else 0
        )

        assignment_metrics = st.columns(4)

        assignment_metrics[0].metric(
            "Total leads",
            f"{len(assigned_leads):,}",
        )

        assignment_metrics[1].metric(
            "Assigned",
            f"{total_assigned:,}",
        )

        assignment_metrics[2].metric(
            "Unassigned",
            f"{total_unassigned:,}",
        )

        assignment_metrics[3].metric(
            "Assignment rate",
            f"{assignment_rate:.1%}",
        )

        assignment_tab, capacity_tab = st.tabs(
            [
                "Lead assignments",
                "Updated representative capacity",
            ]
        )

        with assignment_tab:
            assignment_filter = st.selectbox(
                "Show assignments",
                options=[
                    "All",
                    "Assigned only",
                    "Unassigned only",
                ],
            )

            assignments_to_display = assigned_leads.copy()

            if assignment_filter == "Assigned only":
                assignments_to_display = assignments_to_display[
                    assignments_to_display["Assignment_Status"]
                    == "assigned"
                ]

            elif assignment_filter == "Unassigned only":
                assignments_to_display = assignments_to_display[
                    assignments_to_display["Assignment_Status"]
                    == "unassigned"
                ]

            st.dataframe(
                assignments_to_display,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label="Download lead assignments",
                data=dataframe_to_csv(assigned_leads),
                file_name="lead_assignments.csv",
                mime="text/csv",
            )

        with capacity_tab:
            displayed_updated_reps = updated_reps.copy()

            displayed_updated_reps[
                "Final_Capacity_Utilization"
            ] = (
                displayed_updated_reps[
                    "Final_Capacity_Utilization"
                ]
                .mul(100)
                .round(1)
                .astype("string")
                + "%"
            )

            st.dataframe(
                displayed_updated_reps,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label="Download updated representative capacity",
                data=dataframe_to_csv(updated_reps),
                file_name="updated_representative_capacity.csv",
                mime="text/csv",
            )


# ============================================================
# EMPTY-STATE MESSAGE
# ============================================================

if lead_file is None and rep_file is None:
    st.info(
        "Upload your leads CSV and representatives CSV to begin."
    )
