# ============================================================
# IMPORT LIBRARIES
# ============================================================

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Team 36's Lead Matcher System",
    page_icon="📊",
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
    "Seniority_Level",
    "Current_Capacity",
    "Max_Capacity",
}


# ============================================================
# COLUMN-HEADER ALIASES
# ============================================================

HEADER_ALIASES = {
    # Lead columns
    "lead_id": "Lead_ID",
    "leadid": "Lead_ID",

    "company_name": "Company_Name",
    "companyname": "Company_Name",

    "industry_vertical": "Industry_Vertical",
    "industryvertical": "Industry_Vertical",

    "company_size": "Company_Size",
    "companysize": "Company_Size",

    "recent_activity": "Recent_Activity",
    "recentactivity": "Recent_Activity",

    "vip_referral": "VIP_Referral",
    "vipreferral": "VIP_Referral",

    # Representative columns
    "rep_id": "Rep_ID",
    "repid": "Rep_ID",

    "name": "Name",

    "industry_vertical_expertise":
        "Industry_Vertical_Expertise",

    "industryverticalexpertise":
        "Industry_Vertical_Expertise",

    # Accept different versions of seniority
    "seniority_level": "Seniority_Level",
    "senioritylevel": "Seniority_Level",
    "seniority": "Seniority_Level",

    "current_capacity": "Current_Capacity",
    "currentcapacity": "Current_Capacity",

    "max_capacity": "Max_Capacity",
    "maxcapacity": "Max_Capacity",

    # Shared columns
    "region": "Region",
    "language": "Language",
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

    "northamericaasiapacific":
        "north america and asia pacific",
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
# COLUMN-HEADER CLEANING
# ============================================================

def clean_column_names(dataframe):
    """
    Clean uploaded CSV column names and convert aliases into
    the exact names expected by the application.

    Examples:
        " Seniority Level " -> "Seniority_Level"
        "Seniority_Level]"  -> "Seniority_Level"
        "current capacity"  -> "Current_Capacity"
    """

    dataframe = dataframe.copy()

    cleaned_columns = (
        pd.Index(dataframe.columns)
        .astype("string")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    final_columns = [
        HEADER_ALIASES.get(column, column)
        for column in cleaned_columns
    ]

    duplicate_columns = (
        pd.Index(final_columns)[
            pd.Index(final_columns).duplicated()
        ]
        .unique()
        .tolist()
    )

    if duplicate_columns:
        duplicate_text = ", ".join(duplicate_columns)

        raise ValueError(
            "Duplicate columns were found after cleaning: "
            f"{duplicate_text}"
        )

    dataframe.columns = final_columns

    return dataframe


def require_columns(
    dataframe,
    required_columns,
    file_name,
):
    """Verify that a file contains all required columns."""

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        received_text = ", ".join(
            str(column)
            for column in dataframe.columns
        )

        raise ValueError(
            f"{file_name} is missing: {missing_text}. "
            f"Columns found: {received_text}"
        )


# ============================================================
# GENERAL CLEANING FUNCTIONS
# ============================================================

def normalize_text(series):
    """
    Remove outside spaces, convert to lowercase, and preserve
    missing values.
    """

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
    Create dictionary lookup keys by removing punctuation
    and spaces.

    Example:
        "Asia-Pacific" -> "asiapacific"
    """

    return (
        normalize_text(series)
        .str.replace(
            r"[^a-z0-9]+",
            "",
            regex=True,
        )
    )


def map_normalized(series, mapping):
    """Normalize a column and apply a mapping dictionary."""

    normalized_keys = normalize_key(series)

    return (
        normalized_keys
        .map(mapping)
        .astype("string")
    )


def dataframe_to_csv(dataframe):
    """Convert a DataFrame into downloadable CSV bytes."""

    return (
        dataframe
        .to_csv(index=False)
        .encode("utf-8")
    )


def clean_company_size(series):
    """
    Convert company-size values into nullable integers.

    Examples:
        500   -> 500
        2.5k  -> 2500
        1,200 -> 1200
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

    contains_k = text.str.endswith(
        "k",
        na=False,
    )

    multiplier = np.where(
        contains_k,
        1000,
        1,
    )

    result = number * multiplier

    return result.round().astype("Int64")


def regions_overlap(region_one, region_two):
    """
    Check whether two normalized region values overlap.

    Example:
        "asia pacific and europe" overlaps with "europe"
    """

    if (
        pd.isna(region_one)
        or pd.isna(region_two)
    ):
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

    return bool(
        regions_one.intersection(regions_two)
    )


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
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
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

    vip_text = normalize_text(
        leads["VIP_Referral"]
    )

    leads["VIP_Referral"] = (
        vip_text
        .map({
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
# REPRESENTATIVE CAPACITY CALCULATIONS
# ============================================================

def recalculate_rep_capacity(reps):
    """Recalculate representative capacity information."""

    reps = reps.copy()

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
        .round()
        .astype("Int64")
    )

    reps["Invalid_Max_Capacity"] = (
        reps["Max_Capacity"] <= 0
    )

    current_capacity = (
        reps["Current_Capacity"]
        .astype("Float64")
    )

    max_capacity = (
        reps["Max_Capacity"]
        .astype("Float64")
    )

    reps["Capacity_Utilization"] = (
        current_capacity
        .div(
            max_capacity.where(
                max_capacity > 0
            )
        )
        .fillna(1.0)
        .clip(lower=0)
    )

    reps["Available_Capacity"] = (
        reps["Max_Capacity"]
        - reps["Current_Capacity"]
    ).clip(lower=0)

    return reps


# ============================================================
# REPRESENTATIVE CLEANING
# ============================================================

def clean_reps(reps):
    """Clean and normalize representatives."""

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

    # Example:
    # "Maria Fernandez (West)" -> "Maria Fernandez"
    reps["Name"] = (
        reps["Name"]
        .astype("string")
        .str.replace(
            r"\s*\([^)]*\)\s*",
            " ",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
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

    reps["Industry_Vertical_Expertise"] = (
        normalize_text(
            reps["Industry_Vertical_Expertise"]
        )
    )

    reps["Seniority_Level"] = map_normalized(
        reps["Seniority_Level"],
        SENIORITY_MAP,
    )

    reps = recalculate_rep_capacity(reps)

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
            leads["Company_Size"]
            .ge(1000)
            .fillna(False)
            .to_numpy(dtype=bool),

            leads["Company_Size"]
            .ge(100)
            .fillna(False)
            .to_numpy(dtype=bool),

            leads["Company_Size"]
            .isna()
            .to_numpy(dtype=bool),
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

    leads["Score"] = score.mask(
        is_vip,
        100,
    )

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
        leads
        .sort_values(
            by=[
                "Score",
                "Lead_ID",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(leads, reps):
    """Return a list of data-quality warnings."""

    warnings = []

    if leads["Lead_ID"].isna().any():
        warnings.append(
            "Some leads have missing Lead_ID values."
        )

    duplicate_leads = (
        leads["Lead_ID"]
        .duplicated()
        .sum()
    )

    if duplicate_leads:
        warnings.append(
            f"{duplicate_leads} duplicate Lead_ID "
            "values were found."
        )

    if reps["Rep_ID"].isna().any():
        warnings.append(
            "Some representatives have missing Rep_ID values."
        )

    duplicate_reps = (
        reps["Rep_ID"]
        .duplicated()
        .sum()
    )

    if duplicate_reps:
        warnings.append(
            f"{duplicate_reps} duplicate Rep_ID "
            "values were found."
        )

    over_capacity = (
        reps["Current_Capacity"]
        .gt(reps["Max_Capacity"])
        .fillna(False)
    )

    if over_capacity.any():
        warnings.append(
            f"{over_capacity.sum()} representatives "
            "currently exceed their maximum capacity."
        )

    invalid_maximum = (
        reps["Max_Capacity"] <= 0
    ).sum()

    if invalid_maximum:
        warnings.append(
            f"{invalid_maximum} representatives have "
            "zero or negative maximum capacity."
        )

    missing_lead_regions = (
        leads["Region"]
        .isna()
        .sum()
    )

    if missing_lead_regions:
        warnings.append(
            f"{missing_lead_regions} leads have an "
            "unknown region."
        )

    missing_lead_languages = (
        leads["Language"]
        .isna()
        .sum()
    )

    if missing_lead_languages:
        warnings.append(
            f"{missing_lead_languages} leads have an "
            "unknown language."
        )

    missing_rep_regions = (
        reps["Region"]
        .isna()
        .sum()
    )

    if missing_rep_regions:
        warnings.append(
            f"{missing_rep_regions} representatives "
            "have an unknown region."
        )

    missing_rep_languages = (
        reps["Language"]
        .isna()
        .sum()
    )

    if missing_rep_languages:
        warnings.append(
            f"{missing_rep_languages} representatives "
            "have an unknown language."
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
    Assign leads to representatives based on:

    1. Available capacity
    2. Region
    3. Language
    4. Industry expertise
    5. Seniority
    6. Capacity utilization
    """

    assigned_leads = leads.copy()
    updated_reps = reps.copy()

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

    # Leads are already sorted highest priority first.
    for _, lead in assigned_leads.iterrows():
        best_rep_index = None
        best_ranking = None
        best_match_score = 0

        for rep_index, rep in updated_reps.iterrows():
            current = current_capacity[rep_index]
            maximum = max_capacity[rep_index]

            # Skip representatives with invalid or full capacity.
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
                and pd.notna(
                    rep["Industry_Vertical_Expertise"]
                )
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

            available_capacity = maximum - current

            # Critical and high leads favor senior reps.
            if lead["Priority"] in {
                "critical",
                "high",
            }:
                ranking = (
                    match_score,
                    seniority_score,
                    -utilization,
                    available_capacity,
                )

            else:
                ranking = (
                    match_score,
                    -utilization,
                    available_capacity,
                    seniority_score,
                )

            if (
                best_ranking is None
                or ranking > best_ranking
            ):
                best_ranking = ranking
                best_rep_index = rep_index
                best_match_score = match_score

        if best_rep_index is None:
            assigned_rep_ids.append(pd.NA)
            assigned_rep_names.append(pd.NA)
            assignment_statuses.append("unassigned")
            match_scores.append(0)
            continue

        selected_rep = updated_reps.loc[
            best_rep_index
        ]

        assigned_rep_ids.append(
            selected_rep["Rep_ID"]
        )

        assigned_rep_names.append(
            selected_rep["Name"]
        )

        assignment_statuses.append("assigned")
        match_scores.append(best_match_score)

        # Increase the selected rep's current workload.
        current_capacity[best_rep_index] += 1

    assigned_leads["Assigned_Rep_ID"] = (
        assigned_rep_ids
    )

    assigned_leads["Assigned_Rep_Name"] = (
        assigned_rep_names
    )

    assigned_leads["Assignment_Status"] = (
        assignment_statuses
    )

    assigned_leads["Match_Score"] = match_scores

    updated_reps["Final_Current_Capacity"] = [
        current_capacity[index]
        for index in updated_reps.index
    ]

    updated_reps["Final_Available_Capacity"] = (
        updated_reps["Max_Capacity"]
        - updated_reps["Final_Current_Capacity"]
    ).clip(lower=0)

    updated_reps["Final_Capacity_Utilization"] = (
        np.where(
            updated_reps["Max_Capacity"] > 0,
            (
                updated_reps["Final_Current_Capacity"]
                / updated_reps["Max_Capacity"]
            ),
            1.0,
        )
    )

    updated_reps["New_Leads_Assigned"] = (
        updated_reps["Final_Current_Capacity"]
        - updated_reps["Current_Capacity"]
    )

    updated_reps = (
        updated_reps
        .sort_values(
            by=[
                "Final_Capacity_Utilization",
                "Final_Available_Capacity",
                "Rep_ID",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return assigned_leads, updated_reps


# ============================================================
# CACHED FILE PROCESSING
# ============================================================

@st.cache_data(
    show_spinner="Cleaning and scoring leads..."
)
def process_leads(file_bytes):
    """Read, clean, and score a leads CSV."""

    leads = pd.read_csv(
        io.BytesIO(file_bytes)
    )

    leads = clean_column_names(leads)
    leads = clean_leads(leads)
    leads = score_leads(leads)

    return leads


@st.cache_data(
    show_spinner="Cleaning representatives..."
)
def process_reps(file_bytes):
    """Read, clean, and sort a representatives CSV."""

    reps = pd.read_csv(
        io.BytesIO(file_bytes)
    )

    reps = clean_column_names(reps)
    reps = clean_reps(reps)

    reps = (
        reps
        .sort_values(
            by=[
                "Capacity_Utilization",
                "Available_Capacity",
                "Rep_ID",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return reps


# ============================================================
# STREAMLIT HELPER FUNCTIONS
# ============================================================

def make_file_signature(*uploaded_files):
    """
    Create an identifier for the uploaded files.

    This lets the app clear old assignment results when new
    files are uploaded.
    """

    file_bytes = b""

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_bytes += uploaded_file.getvalue()

    if not file_bytes:
        return None

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def apply_capacity_adjustments(
    reps,
    adjustments,
):
    """Apply user-saved maximum-capacity corrections."""

    reps = reps.copy()

    for rep_id, new_capacity in adjustments.items():
        rep_mask = (
            reps["Rep_ID"] == rep_id
        )

        reps.loc[
            rep_mask,
            "Max_Capacity",
        ] = int(new_capacity)

    return recalculate_rep_capacity(reps)


def create_pie_chart(
    data,
    category_column,
    count_column,
    title,
):
    """Display a pie chart with Vega-Lite."""

    chart_data = data[
        [
            category_column,
            count_column,
        ]
    ].copy()

    st.vega_lite_chart(
        chart_data,
        {
            "title": title,
            "mark": {
                "type": "arc",
                "innerRadius": 45,
                "tooltip": True,
            },
            "encoding": {
                "theta": {
                    "field": count_column,
                    "type": "quantitative",
                },
                "color": {
                    "field": category_column,
                    "type": "nominal",
                    "legend": {
                        "title": category_column,
                    },
                },
                "tooltip": [
                    {
                        "field": category_column,
                        "type": "nominal",
                    },
                    {
                        "field": count_column,
                        "type": "quantitative",
                    },
                ],
            },
        },
        use_container_width=True,
    )


def filter_and_sort_leads(
    leads,
    key_prefix,
):
    """Create lead filtering and sorting controls."""

    filtered = leads.copy()

    filter_columns = st.columns(4)

    with filter_columns[0]:
        priority_filter = st.multiselect(
            "Priority",
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
            key=f"{key_prefix}_priority",
        )

    with filter_columns[1]:
        region_options = sorted(
            filtered["Region"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_region = st.selectbox(
            "Region",
            options=[
                "All",
                *region_options,
            ],
            key=f"{key_prefix}_region",
        )

    with filter_columns[2]:
        language_options = sorted(
            filtered["Language"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_language = st.selectbox(
            "Language",
            options=[
                "All",
                *language_options,
            ],
            key=f"{key_prefix}_language",
        )

    with filter_columns[3]:
        sort_option = st.selectbox(
            "Sort by",
            options=[
                "Priority: highest first",
                "Priority: lowest first",
                "Score: highest first",
                "Score: lowest first",
                "Company size: largest first",
                "Company name: A–Z",
            ],
            key=f"{key_prefix}_sort",
        )

    filtered = filtered[
        filtered["Priority"].isin(
            priority_filter
        )
    ]

    if selected_region != "All":
        filtered = filtered[
            filtered["Region"]
            == selected_region
        ]

    if selected_language != "All":
        filtered = filtered[
            filtered["Language"]
            == selected_language
        ]

    priority_order = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    filtered["_Priority_Rank"] = (
        filtered["Priority"]
        .map(priority_order)
        .fillna(0)
    )

    if sort_option == "Priority: highest first":
        filtered = filtered.sort_values(
            by=[
                "_Priority_Rank",
                "Score",
                "Lead_ID",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )

    elif sort_option == "Priority: lowest first":
        filtered = filtered.sort_values(
            by=[
                "_Priority_Rank",
                "Score",
                "Lead_ID",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            kind="stable",
        )

    elif sort_option == "Score: highest first":
        filtered = filtered.sort_values(
            by=[
                "Score",
                "Lead_ID",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )

    elif sort_option == "Score: lowest first":
        filtered = filtered.sort_values(
            by=[
                "Score",
                "Lead_ID",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )

    elif sort_option == "Company size: largest first":
        filtered = filtered.sort_values(
            by=[
                "Company_Size",
                "Score",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
            na_position="last",
        )

    elif sort_option == "Company name: A–Z":
        filtered = filtered.sort_values(
            by=[
                "Company_Name",
                "Lead_ID",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )

    return (
        filtered
        .drop(
            columns=["_Priority_Rank"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "assigned_leads" not in st.session_state:
    st.session_state.assigned_leads = None

if "updated_reps" not in st.session_state:
    st.session_state.updated_reps = None

if "capacity_adjustments" not in st.session_state:
    st.session_state.capacity_adjustments = {}

if "uploaded_file_signature" not in st.session_state:
    st.session_state.uploaded_file_signature = None


# ============================================================
# STREAMLIT PAGE HEADER
# ============================================================

st.title("Team 36's Lead Matcher System")

st.write(
    """
    Upload lead and representative CSV files to clean, score,
    organize, and assign leads.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Navigation")

    page = st.selectbox(
        "Go to",
        options=[
            "Dashboard",
            "Lead Queue",
            "Assignments",
            "Representative Capacity",
        ],
    )

    st.divider()

    st.header("Assignment Settings")

    matching_option = st.selectbox(
        "Required matching",
        options=[
            "Region and language",
            "Region only",
            "Language only",
            "No required match",
        ],
    )

    if matching_option == "Region and language":
        require_region = True
        require_language = True

    elif matching_option == "Region only":
        require_region = True
        require_language = False

    elif matching_option == "Language only":
        require_region = False
        require_language = True

    else:
        require_region = False
        require_language = False

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


# ============================================================
# FILE UPLOADS
# ============================================================

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

current_signature = make_file_signature(
    lead_file,
    rep_file,
)

# Clear old assignments when the uploaded files change.
if (
    current_signature is not None
    and current_signature
    != st.session_state.uploaded_file_signature
):
    st.session_state.uploaded_file_signature = (
        current_signature
    )

    st.session_state.assigned_leads = None
    st.session_state.updated_reps = None
    st.session_state.capacity_adjustments = {}


# ============================================================
# PROCESS LEADS FILE
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
# PROCESS REPRESENTATIVES FILE
# ============================================================

if rep_file is not None:
    try:
        cleaned_reps = process_reps(
            rep_file.getvalue()
        )

        cleaned_reps = apply_capacity_adjustments(
            cleaned_reps,
            st.session_state.capacity_adjustments,
        )

    except Exception as error:
        st.error(
            "Unable to process the representatives file: "
            f"{error}"
        )


if lead_file is None and rep_file is None:
    st.info(
        "Upload your leads CSV and representatives CSV "
        "to begin."
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":
    st.header("Dashboard")

    if cleaned_leads is not None:
        lead_metrics = st.columns(4)

        lead_metrics[0].metric(
            "Total leads",
            f"{len(cleaned_leads):,}",
        )

        lead_metrics[1].metric(
            "Critical",
            f"{(
                cleaned_leads['Priority'] == 'critical'
            ).sum():,}",
        )

        lead_metrics[2].metric(
            "High priority",
            f"{(
                cleaned_leads['Priority'] == 'high'
            ).sum():,}",
        )

        average_score = (
            cleaned_leads["Score"].mean()
            if not cleaned_leads.empty
            else 0
        )

        lead_metrics[3].metric(
            "Average score",
            f"{average_score:.1f}",
        )

    if cleaned_reps is not None:
        st.subheader("Representative Capacity")

        rep_metrics = st.columns(3)

        rep_metrics[0].metric(
            "Representatives",
            f"{len(cleaned_reps):,}",
        )

        rep_metrics[1].metric(
            "Available capacity",
            f"{cleaned_reps[
                'Available_Capacity'
            ].sum():,.0f}",
        )

        rep_metrics[2].metric(
            "Invalid maximum capacity",
            f"{cleaned_reps[
                'Invalid_Max_Capacity'
            ].sum():,}",
        )

    if st.session_state.assigned_leads is not None:
        assigned_leads = (
            st.session_state.assigned_leads
        )

        assigned_count = (
            assigned_leads["Assignment_Status"]
            .eq("assigned")
            .sum()
        )

        unassigned_count = (
            assigned_leads["Assignment_Status"]
            .eq("unassigned")
            .sum()
        )

        assignment_chart_data = pd.DataFrame({
            "Status": [
                "Assigned",
                "Unassigned",
            ],
            "Count": [
                assigned_count,
                unassigned_count,
            ],
        })

        priority_chart_data = (
            assigned_leads["Priority"]
            .value_counts()
            .rename_axis("Priority")
            .reset_index(name="Count")
        )

        st.subheader("Assignment Charts")

        chart_one, chart_two = st.columns(2)

        with chart_one:
            create_pie_chart(
                assignment_chart_data,
                "Status",
                "Count",
                "Assigned vs. Unassigned Leads",
            )

        with chart_two:
            create_pie_chart(
                priority_chart_data,
                "Priority",
                "Count",
                "Leads by Priority",
            )

    elif cleaned_leads is not None:
        st.info(
            "Run the lead assignment to display the "
            "assigned and unassigned pie chart."
        )


# ============================================================
# LEAD QUEUE PAGE
# ============================================================

elif page == "Lead Queue":
    st.header("Lead Queue")

    if cleaned_leads is None:
        st.info(
            "Upload a leads file to display the queue."
        )

    else:
        if st.session_state.assigned_leads is not None:
            queue = (
                st.session_state.assigned_leads[
                    st.session_state.assigned_leads[
                        "Assignment_Status"
                    ] == "unassigned"
                ]
                .copy()
            )

            st.caption(
                "Showing leads that are still waiting "
                "for assignment."
            )

        else:
            queue = cleaned_leads.copy()

            st.caption(
                "Assignment has not been run, so every "
                "lead is currently in the queue."
            )

        queue = filter_and_sort_leads(
            queue,
            "lead_queue",
        )

        st.metric(
            "Leads waiting",
            f"{len(queue):,}",
        )

        st.dataframe(
            queue,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download lead queue",
            data=dataframe_to_csv(queue),
            file_name="lead_queue.csv",
            mime="text/csv",
        )


# ============================================================
# ASSIGNMENTS PAGE
# ============================================================

elif page == "Assignments":
    st.header("Lead Assignments")

    if (
        cleaned_leads is None
        or cleaned_reps is None
    ):
        st.info(
            "Upload both the leads and representatives "
            "files to run assignment."
        )

    else:
        validation_warnings = validate_data(
            cleaned_leads,
            cleaned_reps,
        )

        if validation_warnings:
            for warning in validation_warnings:
                st.warning(warning)

        else:
            st.success(
                "The uploaded files passed validation."
            )

        invalid_capacity_count = int(
            cleaned_reps[
                "Invalid_Max_Capacity"
            ].sum()
        )

        if invalid_capacity_count:
            st.warning(
                f"{invalid_capacity_count} representatives "
                "have zero or negative maximum capacity. "
                "They cannot receive leads until their "
                "capacity is corrected."
            )

        if st.button(
            "Assign leads to representatives",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Assigning leads..."):
                assigned_leads, updated_reps = (
                    match_and_assign(
                        cleaned_leads,
                        cleaned_reps,
                        require_region=require_region,
                        require_language=require_language,
                    )
                )

            st.session_state.assigned_leads = (
                assigned_leads
            )

            st.session_state.updated_reps = updated_reps

        if st.session_state.assigned_leads is not None:
            assigned_leads = (
                st.session_state.assigned_leads
            )

            updated_reps = (
                st.session_state.updated_reps
            )

            total_assigned = (
                assigned_leads["Assignment_Status"]
                .eq("assigned")
                .sum()
            )

            total_unassigned = (
                assigned_leads["Assignment_Status"]
                .eq("unassigned")
                .sum()
            )

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

            status_filter = st.selectbox(
                "Assignment status",
                options=[
                    "All",
                    "Assigned",
                    "Unassigned",
                ],
            )

            displayed_assignments = (
                assigned_leads.copy()
            )

            if status_filter == "Assigned":
                displayed_assignments = (
                    displayed_assignments[
                        displayed_assignments[
                            "Assignment_Status"
                        ] == "assigned"
                    ]
                )

            elif status_filter == "Unassigned":
                displayed_assignments = (
                    displayed_assignments[
                        displayed_assignments[
                            "Assignment_Status"
                        ] == "unassigned"
                    ]
                )

            summary_columns = [
                "Lead_ID",
                "Company_Name",
                "Assigned_Rep_ID",
                "Assigned_Rep_Name",
            ]

            assignment_summary = (
                displayed_assignments[
                    summary_columns
                ]
                .reset_index(drop=True)
            )

            st.subheader("Assignment Summary")

            st.caption(
                "Select a row to view all information "
                "for that lead."
            )

            selection_event = st.dataframe(
                assignment_summary,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="assignment_summary_table",
            )

            selected_rows = (
                selection_event.selection.rows
            )

            if selected_rows:
                selected_position = selected_rows[0]

                selected_lead_id = (
                    assignment_summary
                    .iloc[selected_position]["Lead_ID"]
                )

                selected_lead = assigned_leads[
                    assigned_leads["Lead_ID"]
                    == selected_lead_id
                ]

                st.subheader("Selected Lead Details")

                st.dataframe(
                    selected_lead,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                label="Download lead assignments",
                data=dataframe_to_csv(
                    assigned_leads
                ),
                file_name="lead_assignments.csv",
                mime="text/csv",
            )

            st.download_button(
                label=(
                    "Download updated representative capacity"
                ),
                data=dataframe_to_csv(
                    updated_reps
                ),
                file_name=(
                    "updated_representative_capacity.csv"
                ),
                mime="text/csv",
            )


# ============================================================
# REPRESENTATIVE CAPACITY PAGE
# ============================================================

elif page == "Representative Capacity":
    st.header("Representative Capacity")

    if cleaned_reps is None:
        st.info(
            "Upload a representatives file to manage "
            "capacity."
        )

    else:
        all_reps_tab, invalid_reps_tab = st.tabs(
            [
                "All Representatives",
                "Fix Invalid Capacity",
            ]
        )

        with all_reps_tab:
            displayed_reps = cleaned_reps.copy()

            displayed_reps[
                "Capacity_Utilization"
            ] = (
                displayed_reps[
                    "Capacity_Utilization"
                ]
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
                label="Download representatives",
                data=dataframe_to_csv(
                    cleaned_reps
                ),
                file_name=(
                    "cleaned_representatives.csv"
                ),
                mime="text/csv",
            )

        with invalid_reps_tab:
            invalid_reps = (
                cleaned_reps[
                    cleaned_reps[
                        "Max_Capacity"
                    ] <= 0
                ]
                .copy()
            )

            if invalid_reps.empty:
                st.success(
                    "No representatives have zero or "
                    "negative maximum capacity."
                )

            else:
                st.warning(
                    f"{len(invalid_reps)} representatives "
                    "need a valid maximum capacity."
                )

                capacity_editor = invalid_reps[
                    [
                        "Rep_ID",
                        "Name",
                        "Current_Capacity",
                        "Max_Capacity",
                    ]
                ].copy()

                capacity_editor = (
                    capacity_editor.rename(
                        columns={
                            "Max_Capacity":
                                "New_Max_Capacity",
                        }
                    )
                )

                # Start every correction at 1.
                capacity_editor[
                    "New_Max_Capacity"
                ] = 1

                edited_capacity = st.data_editor(
                    capacity_editor,
                    use_container_width=True,
                    hide_index=True,
                    disabled=[
                        "Rep_ID",
                        "Name",
                        "Current_Capacity",
                    ],
                    column_config={
                        "New_Max_Capacity":
                            st.column_config.NumberColumn(
                                "New maximum capacity",
                                min_value=1,
                                step=1,
                                required=True,
                            ),
                    },
                    key="capacity_editor",
                )

                if st.button(
                    "Save capacity changes",
                    type="primary",
                ):
                    invalid_values = (
                        edited_capacity[
                            "New_Max_Capacity"
                        ] <= 0
                    )

                    if invalid_values.any():
                        st.error(
                            "Every new maximum capacity "
                            "must be greater than zero."
                        )

                    else:
                        for _, row in (
                            edited_capacity.iterrows()
                        ):
                            rep_id = row["Rep_ID"]

                            new_maximum = int(
                                row[
                                    "New_Max_Capacity"
                                ]
                            )

                            st.session_state[
                                "capacity_adjustments"
                            ][rep_id] = new_maximum

                        # Capacity changes make old
                        # assignment results outdated.
                        st.session_state.assigned_leads = None
                        st.session_state.updated_reps = None

                        st.success(
                            "Capacity changes were saved. "
                            "Run assignment again to use "
                            "the corrected values."
                        )

                        st.rerun()
