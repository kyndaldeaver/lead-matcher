from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lead Matcher",
    layout="wide",
)


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

LEAD_REQUIRED = {
    "Lead_ID",
    "Company_Name",
    "Region",
    "Language",
    "Industry_Vertical",
    "Company_Size",
    "Recent_Activity",
    "VIP_Referral",
}

REP_REQUIRED = {
    "Rep_ID",
    "Name",
    "Region",
    "Language",
    "Industry_Vertical_Expertise",
    "Current_Capacity",
    "Max_Capacity",
}


# ============================================================
# COLUMN HEADER ALIASES
# ============================================================

HEADER_ALIASES = {
    # Lead columns
    "leadid": "Lead_ID",
    "companyname": "Company_Name",
    "industryvertical": "Industry_Vertical",
    "companysize": "Company_Size",
    "recentactivity": "Recent_Activity",
    "vipreferral": "VIP_Referral",

    # Representative columns
    "repid": "Rep_ID",
    "name": "Name",
    "industryverticalexpertise":
        "Industry_Vertical_Expertise",
    "currentcapacity": "Current_Capacity",
    "maxcapacity": "Max_Capacity",

    # Shared columns
    "region": "Region",
    "language": "Language",
}


# ============================================================
# DATA NORMALIZATION DICTIONARIES
# ============================================================

REGION_ALIASES = {
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

    "asiapacificeurope":
        "asia pacific and europe",

    "northamericaasiapacific":
        "north america and asia pacific",
}

LANGUAGE_ALIASES = {
    "en": "english",
    "english": "english",

    "es": "spanish",
    "spanish": "spanish",

    "de": "german",
    "german": "german",
}

ACTIVITY_ALIASES = {
    "na": "N/A",
    "unknown": "Unknown",

    "subscribed":
        "Subscribed To Blog",

    "subscribedtoblog":
        "Subscribed To Blog",

    "downloadedwhitepaper":
        "Downloaded Whitepaper",

    "requestedlivedemo":
        "Requested Live Demo",

    "openedmarketingemail":
        "Opened Marketing Email",

    "contactsales":
        "Contact Sales",

    "visitedpricingpage":
        "Visited Pricing Page",

    "startedfreetrial":
        "Started Free Trial",

    "attendedwebinar":
        "Attended Webinar",
}

ACTIVITY_POINTS = {
    "Requested Live Demo": 35,
    "Contact Sales": 35,

    "Attended Webinar": 25,
    "Started Free Trial": 25,

    "Downloaded Whitepaper": 15,
    "Visited Pricing Page": 10,

    "Subscribed To Blog": 5,
}

INDUSTRY_POINTS = {
    "technology": 15,
    "healthcare": 10,
    "finance": 10,
}

PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


# ============================================================
# COLUMN HEADER CLEANING
# ============================================================

def clean_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize CSV column names.

    Examples:
        Lead ID -> Lead_ID
        Lead-ID -> Lead_ID
        lead_id -> Lead_ID
    """

    data = dataframe.copy()

    original_headers = (
        pd.Index(data.columns)
        .astype("string")
        .str.replace(
            "\ufeff",
            "",
            regex=False,
        )
        .str.strip()
    )

    header_keys = (
        original_headers
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "",
            regex=True,
        )
    )

    cleaned_headers = [
        HEADER_ALIASES.get(
            key,
            original,
        )
        for key, original in zip(
            header_keys,
            original_headers,
        )
    ]

    if len(cleaned_headers) != len(
        set(cleaned_headers)
    ):
        raise ValueError(
            "The CSV contains duplicate columns after "
            "the column headers were cleaned."
        )

    data.columns = cleaned_headers

    return data


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_name: str,
) -> None:
    """Check whether a CSV contains its required columns."""

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        found_text = ", ".join(
            str(column)
            for column in dataframe.columns
        )

        raise ValueError(
            f"{file_name} is missing required columns: "
            f"{missing_text}. Columns found: {found_text}"
        )


# ============================================================
# GENERAL COLUMN CLEANING FUNCTIONS
# ============================================================

def normalize_text_column(
    series: pd.Series,
) -> pd.Series:
    """
    Clean a complete pandas text column.

    This:
    - Preserves missing values
    - Removes outside spaces
    - Converts text to lowercase
    - Replaces repeated spaces with one space
    """

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
        })
    )


def normalize_key_column(
    series: pd.Series,
) -> pd.Series:
    """
    Prepare an entire column for dictionary matching.

    Examples:
        Asia-Pacific -> asiapacific
        Asia Pacific -> asiapacific
        Asia.Pacific -> asiapacific
    """

    return (
        normalize_text_column(series)
        .str.replace(
            r"[^a-z0-9]+",
            "",
            regex=True,
        )
    )


def clean_mapped_column(
    series: pd.Series,
    mapping: dict[str, str],
    default: str,
    preserve_unknown: bool = False,
) -> pd.Series:
    """
    Clean an entire column and apply a normalization
    dictionary.
    """

    cleaned_text = normalize_text_column(
        series
    )

    cleaned_keys = normalize_key_column(
        series
    )

    mapped_values = cleaned_keys.map(
        mapping
    )

    if preserve_unknown:
        mapped_values = mapped_values.fillna(
            cleaned_text
        )

    return (
        mapped_values
        .fillna(default)
        .astype("string")
    )


def clean_text_value(value: Any) -> str:
    """
    Clean one value for assignment comparisons.

    The assignment process is sequential, so individual
    values must be compared while capacity is changing.
    """

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


# ============================================================
# COMPANY SIZE CLEANING
# ============================================================

def clean_company_size_column(
    series: pd.Series,
) -> pd.Series:
    """
    Clean the complete Company_Size column.

    Examples:
        500   -> 500
        2.5k  -> 2500
        1,200 -> 1200
        N/A   -> missing
    """

    text = (
        normalize_text_column(series)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )

    extracted_numbers = (
        text.str.extract(
            r"([0-9]*\.?[0-9]+)",
            expand=False,
        )
    )

    numbers = pd.to_numeric(
        extracted_numbers,
        errors="coerce",
    )

    multipliers = np.where(
        text.str.endswith(
            "k",
            na=False,
        ),
        1000,
        1,
    )

    result = numbers * multipliers

    return (
        result
        .round()
        .astype("Float64")
    )


# ============================================================
# COMPANY SIZE FILTER GROUPS
# ============================================================

def company_size_groups(
    series: pd.Series,
) -> pd.Series:
    """
    Categorize the entire Company_Size column for filtering.
    """

    conditions = [
        series.isna(),
        series.lt(100).fillna(False),
        series.lt(1000).fillna(False),
    ]

    choices = [
        "Missing",
        "Under 100",
        "100–999",
    ]

    groups = np.select(
        conditions,
        choices,
        default="1,000+",
    )

    return pd.Series(
        groups,
        index=series.index,
        dtype="string",
    )


# ============================================================
# LEAD CLEANING
# ============================================================

def normalize_leads(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the leads table using pandas columns."""

    data = dataframe.copy()

    validate_required_columns(
        data,
        LEAD_REQUIRED,
        "Leads CSV",
    )

    # Clean the complete Lead_ID column.
    data["Lead_ID"] = (
        data["Lead_ID"]
        .astype("string")
        .str.strip()
    )

    # Clean the complete Company_Name column.
    data["Company_Name"] = (
        data["Company_Name"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    # Standardize the complete Region column.
    data["Region"] = clean_mapped_column(
        data["Region"],
        REGION_ALIASES,
        default="N/A",
        preserve_unknown=True,
    )

    # Standardize the complete Language column.
    data["Language"] = clean_mapped_column(
        data["Language"],
        LANGUAGE_ALIASES,
        default="N/A",
        preserve_unknown=False,
    )

    # Clean the complete industry column.
    data["Industry_Vertical"] = (
        normalize_text_column(
            data["Industry_Vertical"]
        )
        .fillna("unknown")
        .astype("string")
    )

    # Convert the complete company-size column to numbers.
    data["Company_Size"] = (
        clean_company_size_column(
            data["Company_Size"]
        )
    )

    # Standardize the complete activity column.
    data["Recent_Activity"] = (
        clean_mapped_column(
            data["Recent_Activity"],
            ACTIVITY_ALIASES,
            default="Unknown",
            preserve_unknown=True,
        )
        .str.title()
    )

    # Standardize the complete VIP column.
    vip_values = normalize_text_column(
        data["VIP_Referral"]
    )

    data["VIP_Referral"] = (
        vip_values
        .where(
            vip_values.isin({
                "true",
                "false",
                "unknown",
            }),
            "unknown",
        )
        .fillna("unknown")
        .astype("string")
    )

    return data


# ============================================================
# REPRESENTATIVE CLEANING
# ============================================================

def normalize_reps(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the representative table using pandas columns."""

    data = dataframe.copy()

    validate_required_columns(
        data,
        REP_REQUIRED,
        "Sales representatives CSV",
    )

    # Clean all representative IDs.
    data["Rep_ID"] = (
        data["Rep_ID"]
        .astype("string")
        .str.strip()
    )

    # Remove parenthetical information from every name.
    #
    # Example:
    # Maria Fernandez (West) -> Maria Fernandez
    data["Name"] = (
        data["Name"]
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
        .fillna("")
    )

    # Standardize all representative regions.
    data["Region"] = clean_mapped_column(
        data["Region"],
        REGION_ALIASES,
        default="N/A",
        preserve_unknown=True,
    )

    # Standardize all representative languages.
    data["Language"] = clean_mapped_column(
        data["Language"],
        LANGUAGE_ALIASES,
        default="N/A",
        preserve_unknown=False,
    )

    # Clean all industry-expertise values.
    data["Industry_Vertical_Expertise"] = (
        normalize_text_column(
            data["Industry_Vertical_Expertise"]
        )
        .fillna("unknown")
        .astype("string")
    )

    # Convert the complete current-capacity column to numbers.
    data["Current_Capacity"] = (
        pd.to_numeric(
            data["Current_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype("Int64")
    )

    # Convert the complete maximum-capacity column to numbers.
    data["Max_Capacity"] = (
        pd.to_numeric(
            data["Max_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype("Int64")
    )

    maximum = (
        data["Max_Capacity"]
        .astype("Float64")
    )

    current = (
        data["Current_Capacity"]
        .astype("Float64")
    )

    # Calculate utilization for the complete table.
    data["Capacity_Utilization"] = (
        current
        .div(
            maximum.where(
                maximum > 0
            )
        )
        .fillna(1.0)
        .clip(lower=0)
    )

    # Calculate remaining capacity for the complete table.
    data["Available_Capacity"] = (
        data["Max_Capacity"]
        - data["Current_Capacity"]
    ).clip(lower=0)

    return data


# ============================================================
# LEAD SCORING AND SORTING
# ============================================================

def score_leads(
    leads: pd.DataFrame,
) -> pd.DataFrame:
    """Score and sort all leads using pandas columns."""

    data = leads.copy()

    score = pd.Series(
        0,
        index=data.index,
        dtype="int64",
    )

    # --------------------------------------------------------
    # COMPANY SIZE POINTS
    # --------------------------------------------------------

    company_points = np.select(
        [
            data["Company_Size"]
            .ge(1000)
            .fillna(False)
            .to_numpy(dtype=bool),

            data["Company_Size"]
            .ge(100)
            .fillna(False)
            .to_numpy(dtype=bool),

            data["Company_Size"]
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

    score += company_points

    # --------------------------------------------------------
    # ACTIVITY POINTS
    # --------------------------------------------------------

    score += (
        data["Recent_Activity"]
        .map(ACTIVITY_POINTS)
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # INDUSTRY POINTS
    # --------------------------------------------------------

    score += (
        data["Industry_Vertical"]
        .map(INDUSTRY_POINTS)
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # VIP OVERRIDE
    # --------------------------------------------------------

    vip_mask = (
        data["VIP_Referral"]
        .eq("true")
        .fillna(False)
    )

    data["Score"] = score.mask(
        vip_mask,
        100,
    )

    # --------------------------------------------------------
    # PRIORITY CATEGORIES
    # --------------------------------------------------------

    data["Priority"] = np.select(
        [
            data["Score"].ge(85),
            data["Score"].ge(60),
            data["Score"].ge(40),
        ],
        [
            "critical",
            "high",
            "medium",
        ],
        default="low",
    )

    # Create a temporary numeric priority column for sorting.
    data["_Priority_Rank"] = (
        data["Priority"]
        .map(PRIORITY_RANK)
        .fillna(0)
    )

    # Sort by priority, score, and lead ID.
    data = (
        data.sort_values(
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
        .drop(
            columns=["_Priority_Rank"]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# REPRESENTATIVE CAPACITY
# ============================================================

def capacity_ratio(
    current_capacity: float,
    max_capacity: float,
) -> float:
    """Calculate one rep's capacity utilization."""

    if max_capacity <= 0:
        return 1.0

    return min(
        current_capacity / max_capacity,
        1.0,
    )


# ============================================================
# REGION MATCHING
# ============================================================

def regions_overlap(
    lead_region: Any,
    rep_region: Any,
) -> bool:
    """
    Check whether two region values overlap.

    Example:
        europe matches asia pacific and europe
    """

    lead_value = clean_text_value(
        lead_region
    )

    rep_value = clean_text_value(
        rep_region
    )

    invalid_values = {
        "",
        "n/a",
        "unknown",
        "none",
        "nan",
    }

    if (
        lead_value in invalid_values
        or rep_value in invalid_values
    ):
        return False

    lead_regions = {
        region.strip()
        for region in lead_value.split(" and ")
        if region.strip()
    }

    rep_regions = {
        region.strip()
        for region in rep_value.split(" and ")
        if region.strip()
    }

    return bool(
        lead_regions.intersection(
            rep_regions
        )
    )


# ============================================================
# LEAD MATCHING AND ASSIGNMENT
# ============================================================

def match_and_assign(
    leads: pd.DataFrame,
    reps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign leads in descending priority order.

    Matching is sequential because assigning one lead changes
    the representative's capacity for the next lead.
    """

    # Score and sort leads before assignment.
    ranked = score_leads(
        leads
    )

    roster = (
        reps.copy()
        .reset_index(drop=True)
    )

    assignments: list[
        dict[str, Any]
    ] = []

    for _, lead in ranked.iterrows():
        candidates = []

        for rep_index, rep in roster.iterrows():
            max_capacity = float(
                rep["Max_Capacity"]
            )

            current_capacity = float(
                rep["Current_Capacity"]
            )

            # Skip reps with no remaining capacity.
            if (
                max_capacity <= 0
                or current_capacity >= max_capacity
            ):
                continue

            matched_fields = []

            # Region match
            if regions_overlap(
                lead["Region"],
                rep["Region"],
            ):
                matched_fields.append(
                    "region"
                )

            # Language match
            lead_language = clean_text_value(
                lead["Language"]
            )

            rep_language = clean_text_value(
                rep["Language"]
            )

            invalid_values = {
                "",
                "n/a",
                "unknown",
                "none",
                "nan",
            }

            if (
                lead_language not in invalid_values
                and rep_language not in invalid_values
                and lead_language == rep_language
            ):
                matched_fields.append(
                    "language"
                )

            # Industry match
            lead_industry = clean_text_value(
                lead["Industry_Vertical"]
            )

            rep_industry = clean_text_value(
                rep[
                    "Industry_Vertical_Expertise"
                ]
            )

            if (
                lead_industry not in invalid_values
                and rep_industry not in invalid_values
                and lead_industry == rep_industry
            ):
                matched_fields.append(
                    "industry"
                )

            utilization = capacity_ratio(
                current_capacity,
                max_capacity,
            )

            remaining_capacity = (
                max_capacity
                - current_capacity
            )

            # Larger tuple values are preferred.
            ranking = (
                len(matched_fields),
                -utilization,
                remaining_capacity,
                -rep_index,
            )

            candidates.append(
                (
                    ranking,
                    rep_index,
                    matched_fields,
                )
            )

        if candidates:
            (
                _,
                chosen_index,
                matched_fields,
            ) = max(
                candidates,
                key=lambda item: item[0],
            )

            chosen = roster.loc[
                chosen_index
            ]

            roster.at[
                chosen_index,
                "Current_Capacity",
            ] = (
                int(
                    chosen["Current_Capacity"]
                )
                + 1
            )

            assignments.append({
                "Assigned_Rep_ID":
                    chosen["Rep_ID"],

                "Assigned_Rep":
                    chosen["Name"],

                "Rep_Match": (
                    ", ".join(matched_fields)
                    if matched_fields
                    else "No exact attribute match"
                ),

                "Rep_Capacity_Status":
                    "Assigned within capacity",
            })

        else:
            assignments.append({
                "Assigned_Rep_ID": "",
                "Assigned_Rep": "",
                "Rep_Match": "",
                "Rep_Capacity_Status":
                    "No rep capacity available",
            })

    result = pd.concat(
        [
            ranked.reset_index(drop=True),
            pd.DataFrame(assignments),
        ],
        axis=1,
    )

    # Sort the final lead results.
    result["_Priority_Rank"] = (
        result["Priority"]
        .map(PRIORITY_RANK)
        .fillna(0)
    )

    result = (
        result.sort_values(
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
        .drop(
            columns=["_Priority_Rank"]
        )
        .reset_index(drop=True)
    )

    # Recalculate representative capacity by columns.
    roster["Current_Capacity"] = (
        pd.to_numeric(
            roster["Current_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .round()
        .astype("Int64")
    )

    maximum = (
        roster["Max_Capacity"]
        .astype("Float64")
    )

    current = (
        roster["Current_Capacity"]
        .astype("Float64")
    )

    roster["Capacity_Utilization"] = (
        current
        .div(
            maximum.where(
                maximum > 0
            )
        )
        .fillna(1.0)
        .clip(lower=0)
    )

    roster["Available_Capacity"] = (
        roster["Max_Capacity"]
        - roster["Current_Capacity"]
    ).clip(lower=0)

    # Sort representatives by lowest utilization first.
    roster = (
        roster.sort_values(
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

    return result, roster


# ============================================================
# DOWNLOAD HELPER
# ============================================================

def csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """Convert a DataFrame into downloadable CSV bytes."""

    return dataframe.to_csv(
        index=False,
    ).encode("utf-8-sig")


# ============================================================
# CACHED FILE PROCESSING
# ============================================================

@st.cache_data(
    show_spinner="Cleaning lead data..."
)
def process_lead_file(
    file_bytes: bytes,
) -> pd.DataFrame:
    """Read and clean a leads CSV with pandas."""

    data = pd.read_csv(
        pd.io.common.BytesIO(
            file_bytes
        )
    )

    data = clean_column_names(
        data
    )

    data = normalize_leads(
        data
    )

    return data


@st.cache_data(
    show_spinner="Cleaning representative data..."
)
def process_rep_file(
    file_bytes: bytes,
) -> pd.DataFrame:
    """Read and clean a representatives CSV with pandas."""

    data = pd.read_csv(
        pd.io.common.BytesIO(
            file_bytes
        )
    )

    data = clean_column_names(
        data
    )

    data = normalize_reps(
        data
    )

    return data


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("Lead Matcher")

st.write(
    "Upload lead and sales-representative CSV files to clean, "
    "prioritize, and assign leads."
)


# ============================================================
# FILE UPLOAD
# ============================================================

with st.sidebar:
    st.header("Upload files")

    lead_file = st.file_uploader(
        "Leads CSV",
        type=["csv"],
        key="leads",
    )

    rep_file = st.file_uploader(
        "Sales reps CSV",
        type=["csv"],
        key="reps",
    )

    st.caption(
        "Upload separate CSV files with header rows. "
        "See README for required columns."
    )


if not lead_file or not rep_file:
    st.info(
        "Upload both CSV files to start matching."
    )
    st.stop()


# ============================================================
# READ AND CLEAN FILES WITH PANDAS
# ============================================================

try:
    leads_clean = process_lead_file(
        lead_file.getvalue()
    )

    reps_clean = process_rep_file(
        rep_file.getvalue()
    )

except (
    UnicodeDecodeError,
    pd.errors.ParserError,
    ValueError,
    KeyError,
) as error:
    st.error(
        f"Could not process one of the CSV files: {error}"
    )
    st.stop()


if leads_clean.empty or reps_clean.empty:
    st.error(
        "Both files must contain at least one data row."
    )
    st.stop()


# ============================================================
# SCORE AND ASSIGN
# ============================================================

matched, final_roster = match_and_assign(
    leads_clean,
    reps_clean,
)


# ============================================================
# SUMMARY
# ============================================================

st.success(
    f"Processed {len(matched):,} leads and "
    f"{len(final_roster):,} sales reps."
)

assigned_mask = (
    matched["Assigned_Rep_ID"]
    .fillna("")
    .astype("string")
    .str.strip()
    .ne("")
)

assigned_count = int(
    assigned_mask.sum()
)

unassigned_count = int(
    (~assigned_mask).sum()
)

critical_count = int(
    matched["Priority"]
    .eq("critical")
    .sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Leads",
    f"{len(matched):,}",
)

col2.metric(
    "Critical priority",
    f"{critical_count:,}",
)

col3.metric(
    "Assigned",
    f"{assigned_count:,}",
)

col4.metric(
    "Waiting for a rep",
    f"{unassigned_count:,}",
)


# ============================================================
# RESULT TABS
# ============================================================

results_tab, queue_tab = st.tabs(
    [
        "All leads",
        f"Assignment queue ({unassigned_count})",
    ]
)


# ============================================================
# ALL LEADS TAB
# ============================================================

with results_tab:
    st.subheader(
        "Filter and sort leads"
    )

    tracker_col, chart_col = st.columns(
        [2, 1]
    )

    with tracker_col:
        filter1, filter2, filter3 = st.columns(3)

        # ----------------------------------------------------
        # LANGUAGE FILTER
        # ----------------------------------------------------

        with filter1:
            languages = sorted(
                matched["Language"]
                .fillna("N/A")
                .astype(str)
                .unique()
                .tolist()
            )

            language_filter = st.selectbox(
                "Language",
                ["All"] + languages,
            )

        # ----------------------------------------------------
        # INDUSTRY FILTER
        # ----------------------------------------------------

        with filter2:
            industries = sorted(
                matched["Industry_Vertical"]
                .fillna("unknown")
                .astype(str)
                .unique()
                .tolist()
            )

            industry_filter = st.selectbox(
                "Industry",
                ["All"] + industries,
            )

        # ----------------------------------------------------
        # COMPANY SIZE FILTER
        # ----------------------------------------------------

        with filter3:
            all_size_groups = company_size_groups(
                matched["Company_Size"]
            )

            size_options = [
                "Under 100",
                "100–999",
                "1,000+",
                "Missing",
            ]

            available_sizes = [
                group
                for group in size_options
                if group in all_size_groups.values
            ]

            size_filter = st.selectbox(
                "Company size",
                ["All"] + available_sizes,
            )

        # ----------------------------------------------------
        # APPLY COLUMN FILTERS
        # ----------------------------------------------------

        filtered = matched.copy()

        if language_filter != "All":
            filtered = filtered.loc[
                filtered["Language"].eq(
                    language_filter
                )
            ]

        if industry_filter != "All":
            filtered = filtered.loc[
                filtered[
                    "Industry_Vertical"
                ].eq(
                    industry_filter
                )
            ]

        if size_filter != "All":
            filtered_size_groups = (
                company_size_groups(
                    filtered["Company_Size"]
                )
            )

            filtered = filtered.loc[
                filtered_size_groups.eq(
                    size_filter
                )
            ]

        # ----------------------------------------------------
        # SORTING MENU
        # ----------------------------------------------------

        sort_by = st.selectbox(
            "Sort by",
            [
                "Priority score",
                "Language",
                "Industry",
                "Company size",
            ],
        )

        # ----------------------------------------------------
        # APPLY COLUMN SORTING
        # ----------------------------------------------------

        if sort_by == "Company size":
            filtered = filtered.sort_values(
                by=[
                    "Company_Size",
                    "Score",
                    "Lead_ID",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                na_position="last",
                kind="stable",
            )

        elif sort_by == "Priority score":
            filtered = filtered.assign(
                _Priority_Rank=(
                    filtered["Priority"]
                    .map(PRIORITY_RANK)
                    .fillna(0)
                )
            )

            filtered = (
                filtered.sort_values(
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
                .drop(
                    columns=["_Priority_Rank"]
                )
            )

        elif sort_by == "Language":
            filtered = filtered.sort_values(
                by=[
                    "Language",
                    "Score",
                    "Lead_ID",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
                na_position="last",
                kind="stable",
            )

        elif sort_by == "Industry":
            filtered = filtered.sort_values(
                by=[
                    "Industry_Vertical",
                    "Score",
                    "Lead_ID",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
                na_position="last",
                kind="stable",
            )

        filtered = filtered.reset_index(
            drop=True
        )

        st.caption(
            f"Showing {len(filtered):,} "
            f"of {len(matched):,} leads."
        )

        # ----------------------------------------------------
        # LEAD PREVIEW TABLE
        # ----------------------------------------------------

        lead_preview = filtered[
            [
                "Lead_ID",
                "Company_Name",
            ]
        ].copy()

        st.dataframe(
            lead_preview,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # SELECTED LEAD DETAILS
        # ----------------------------------------------------

        if not lead_preview.empty:
            lead_options = (
                lead_preview["Lead_ID"]
                .astype(str)
                .tolist()
            )

            selected_lead_id = st.selectbox(
                "Select a lead to see its full details",
                lead_options,
                key="lead_details_selector",
            )

            selected_lead = filtered.loc[
                filtered["Lead_ID"]
                .astype(str)
                .eq(selected_lead_id)
            ]

            if not selected_lead.empty:
                with st.expander(
                    "Selected lead details"
                ):
                    st.dataframe(
                        selected_lead,
                        use_container_width=True,
                        hide_index=True,
                    )

    # ========================================================
    # ASSIGNMENT PIE CHART
    # ========================================================

    displayed_assigned = int(
        filtered["Assigned_Rep_ID"]
        .fillna("")
        .astype("string")
        .str.strip()
        .ne("")
        .sum()
    )

    assignment_chart = pd.DataFrame({
        "Assignment status": [
            "Assigned",
            "Unassigned",
        ],
        "Leads": [
            displayed_assigned,
            len(filtered) - displayed_assigned,
        ],
    })

    with chart_col:
        st.markdown(
            "<h3 style='text-align: center;'>"
            "Assigned vs. Unassigned Leads"
            "</h3>",
            unsafe_allow_html=True,
        )

        chart_left, chart_center, chart_right = (
            st.columns([1, 4, 1])
        )

        with chart_center:
            st.vega_lite_chart(
                assignment_chart,
                {
                    "width": 260,
                    "height": 300,
                    "mark": {
                        "type": "arc",
                        "outerRadius": 105,
                    },
                    "encoding": {
                        "theta": {
                            "field": "Leads",
                            "type": "quantitative",
                        },
                        "color": {
                            "field":
                                "Assignment status",
                            "type": "nominal",
                            "scale": {
                                "domain": [
                                    "Assigned",
                                    "Unassigned",
                                ],
                                "range": [
                                    "#2E8B57",
                                    "#F4A261",
                                ],
                            },
                            "legend": {
                                "orient": "bottom",
                                "title": None,
                            },
                        },
                        "tooltip": [
                            {
                                "field":
                                    "Assignment status",
                                "type": "nominal",
                            },
                            {
                                "field": "Leads",
                                "type": "quantitative",
                            },
                        ],
                    },
                    "view": {
                        "stroke": None,
                    },
                },
                use_container_width=True,
            )

        # ====================================================
        # LEADS BY SALES REPRESENTATIVE
        # ====================================================

        assigned_rows = filtered.loc[
            filtered["Assigned_Rep_ID"]
            .fillna("")
            .astype("string")
            .str.strip()
            .ne("")
        ].copy()

        assigned_rows["_Rep_ID_Key"] = (
            assigned_rows["Assigned_Rep_ID"]
            .astype("string")
            .str.strip()
        )

        rep_lead_counts = (
            assigned_rows
            .groupby("_Rep_ID_Key")
            .size()
            .rename("Assigned Leads")
        )

        rep_summary = final_roster[
            [
                "Rep_ID",
                "Name",
            ]
        ].copy()

        rep_summary["_Rep_ID_Key"] = (
            rep_summary["Rep_ID"]
            .astype("string")
            .str.strip()
        )

        rep_summary["Assigned Leads"] = (
            rep_summary["_Rep_ID_Key"]
            .map(rep_lead_counts)
            .fillna(0)
            .astype("int64")
        )

        rep_summary = (
            rep_summary[
                [
                    "Rep_ID",
                    "Name",
                    "Assigned Leads",
                ]
            ]
            .sort_values(
                by=[
                    "Assigned Leads",
                    "Name",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        st.subheader(
            "Leads by Sales Rep"
        )

        st.dataframe(
            rep_summary[
                [
                    "Rep_ID",
                    "Name",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not rep_summary.empty:
            rep_options = {
                f"{row.Name} (ID: {row.Rep_ID})":
                    str(row.Rep_ID)
                for row in rep_summary.itertuples(
                    index=False
                )
            }

            selected_rep_label = st.selectbox(
                "Select a rep to see lead count "
                "and full details",
                list(rep_options),
                key="rep_details_selector",
            )

            selected_rep_id = rep_options[
                selected_rep_label
            ]

            selected_rep = final_roster.loc[
                final_roster["Rep_ID"]
                .astype(str)
                .eq(selected_rep_id)
            ]

            selected_rep_count = rep_summary.loc[
                rep_summary["Rep_ID"]
                .astype(str)
                .eq(selected_rep_id),
                "Assigned Leads",
            ]

            if not selected_rep.empty:
                displayed_rep = (
                    selected_rep.copy()
                )

                displayed_rep[
                    "Assigned Leads (current filters)"
                ] = int(
                    selected_rep_count.iloc[0]
                )

                with st.expander(
                    "Selected rep details"
                ):
                    st.dataframe(
                        displayed_rep,
                        use_container_width=True,
                        hide_index=True,
                    )

    # ========================================================
    # DOWNLOAD BUTTONS
    # ========================================================

    st.download_button(
        "Download displayed leads and assignments",
        data=csv_bytes(filtered),
        file_name="leads_with_assignments.csv",
        mime="text/csv",
        key="download_filtered_leads",
    )

    st.download_button(
        "Download updated rep capacities",
        data=csv_bytes(final_roster),
        file_name="sales_reps_updated_capacity.csv",
        mime="text/csv",
        key="download_rep_capacities",
    )


# ============================================================
# ASSIGNMENT QUEUE TAB
# ============================================================

with queue_tab:
    st.subheader(
        "Leads waiting to be assigned"
    )

    queue = (
        matched.loc[
            ~assigned_mask
        ]
        .copy()
    )

    # Guarantee descending priority order.
    if not queue.empty:
        queue["_Priority_Rank"] = (
            queue["Priority"]
            .map(PRIORITY_RANK)
            .fillna(0)
        )

        queue = (
            queue.sort_values(
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
            .drop(
                columns=["_Priority_Rank"]
            )
            .reset_index(drop=True)
        )

    if queue.empty:
        st.success(
            "There are no leads waiting for "
            "a representative."
        )

    else:
        queue.insert(
            0,
            "Queue Position",
            range(
                1,
                len(queue) + 1,
            ),
        )

        st.caption(
            "Queue order follows lead priority score, "
            "highest first. These leads could not be assigned "
            "because no representative had remaining capacity."
        )

        st.dataframe(
            queue,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download assignment queue",
            data=csv_bytes(queue),
            file_name="unassigned_lead_queue.csv",
            mime="text/csv",
            key="download_assignment_queue",
        )


# ============================================================
# RULES EXPLANATION
# ============================================================

with st.expander(
    "Scoring and assignment rules"
):
    st.markdown(
        "- VIP referrals score **100** and are always "
        "**critical**.\n"
        "- Company size: **50** points for 1,000+, "
        "**30** for 100–999, **0** below 100, or "
        "**20** when missing.\n"
        "- Recent activity: demo/contact sales **35**, "
        "webinar/trial **25**, whitepaper **15**, "
        "pricing page **10**, blog subscription **5**.\n"
        "- Industry: technology **15**, "
        "healthcare/finance **10**.\n"
        "- Other priority bands: high **60–84**, "
        "medium **40–59**, low **0–39**.\n"
        "- Rep assignment is a recommendation: region, "
        "language, and industry matches rank first; ties "
        "prefer lower capacity utilization. Each rep is "
        "assigned no more than their positive Max_Capacity."
    )
