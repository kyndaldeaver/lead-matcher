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
# COLUMN HEADER CLEANING
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


def header_key(value: Any) -> str:
    """
    Convert a header into a simple lookup key.

    Examples:
        "Lead ID"   -> "leadid"
        "Lead_ID"   -> "leadid"
        " Lead-ID " -> "leadid"
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def clean_column_names(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize uploaded CSV column names."""

    data = frame.copy()

    cleaned_columns = []

    for column in data.columns:
        key = header_key(column)

        cleaned_columns.append(
            HEADER_ALIASES.get(
                key,
                str(column).strip(),
            )
        )

    if len(cleaned_columns) != len(set(cleaned_columns)):
        raise ValueError(
            "The CSV contains duplicate columns after "
            "cleaning its headers."
        )

    data.columns = cleaned_columns

    return data


# ============================================================
# NORMALIZATION DICTIONARIES
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
    "es": "spanish",
    "spanish": "spanish",

    "en": "english",
    "english": "english",

    "de": "german",
    "german": "german",
}

ACTIVITY_ALIASES = {
    "na": "N/A",
    "unknown": "Unknown",

    "subscribed": "Subscribed To Blog",
    "subscribedtoblog": "Subscribed To Blog",

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

PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


# ============================================================
# GENERAL TEXT CLEANING
# ============================================================

def normalize_text_series(
    series: pd.Series,
) -> pd.Series:
    """
    Clean an entire text column.

    This:
    - Preserves missing values
    - Removes outside spaces
    - Changes text to lowercase
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


def normalize_key_series(
    series: pd.Series,
) -> pd.Series:
    """
    Remove spaces and punctuation from an entire column.

    Examples:
        "Asia-Pacific" -> "asiapacific"
        "Asia Pacific" -> "asiapacific"
        "Asia.Pacific" -> "asiapacific"
    """

    return (
        normalize_text_series(series)
        .str.replace(
            r"[^a-z0-9]+",
            "",
            regex=True,
        )
    )


def clean_text(value: Any) -> str:
    """Clean one text value for comparisons."""

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

def clean_company_size_series(
    series: pd.Series,
) -> pd.Series:
    """
    Clean the complete Company_Size column.

    Examples:
        "500"   -> 500
        "2.5k"  -> 2500
        "1,200" -> 1200
        "N/A"   -> missing
    """

    text = (
        normalize_text_series(series)
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

    numbers = pd.to_numeric(
        text.str.extract(
            r"([0-9]*\.?[0-9]+)",
            expand=False,
        ),
        errors="coerce",
    )

    multiplier = np.where(
        text.str.endswith(
            "k",
            na=False,
        ),
        1000,
        1,
    )

    result = numbers * multiplier

    return (
        result
        .round()
        .astype("Float64")
    )


# ============================================================
# LEAD DATA CLEANING
# ============================================================

def normalize_leads(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and normalize all lead columns."""

    data = frame.copy()

    # Clean lead IDs.
    data["Lead_ID"] = (
        data["Lead_ID"]
        .astype("string")
        .str.strip()
    )

    # Clean company names without making them lowercase.
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

    # Normalize regions.
    original_regions = normalize_text_series(
        data["Region"]
    )

    region_keys = normalize_key_series(
        data["Region"]
    )

    data["Region"] = (
        region_keys
        .map(REGION_ALIASES)
        .fillna(original_regions)
        .fillna("N/A")
        .astype("string")
    )

    # Normalize languages.
    language_keys = normalize_key_series(
        data["Language"]
    )

    data["Language"] = (
        language_keys
        .map(LANGUAGE_ALIASES)
        .fillna("N/A")
        .astype("string")
    )

    # Normalize industry.
    data["Industry_Vertical"] = (
        normalize_text_series(
            data["Industry_Vertical"]
        )
        .fillna("unknown")
        .astype("string")
    )

    # Convert company size to numeric values.
    data["Company_Size"] = (
        clean_company_size_series(
            data["Company_Size"]
        )
    )

    # Normalize recent activity.
    activity_keys = normalize_key_series(
        data["Recent_Activity"]
    )

    original_activity = normalize_text_series(
        data["Recent_Activity"]
    )

    data["Recent_Activity"] = (
        activity_keys
        .map(ACTIVITY_ALIASES)
        .fillna(
            original_activity.str.title()
        )
        .fillna("Unknown")
        .astype("string")
    )

    # Normalize VIP values.
    vip_text = normalize_text_series(
        data["VIP_Referral"]
    )

    data["VIP_Referral"] = (
        vip_text
        .where(
            vip_text.isin({
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
# REPRESENTATIVE DATA CLEANING
# ============================================================

def normalize_reps(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and normalize all representative columns."""

    data = frame.copy()

    # Clean representative IDs.
    data["Rep_ID"] = (
        data["Rep_ID"]
        .astype("string")
        .str.strip()
    )

    # Remove parenthetical information from names.
    #
    # Example:
    # "Maria Fernandez (West)" -> "Maria Fernandez"
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

    # Normalize regions.
    original_regions = normalize_text_series(
        data["Region"]
    )

    region_keys = normalize_key_series(
        data["Region"]
    )

    data["Region"] = (
        region_keys
        .map(REGION_ALIASES)
        .fillna(original_regions)
        .fillna("N/A")
        .astype("string")
    )

    # Normalize languages.
    language_keys = normalize_key_series(
        data["Language"]
    )

    data["Language"] = (
        language_keys
        .map(LANGUAGE_ALIASES)
        .fillna("N/A")
        .astype("string")
    )

    # Normalize industry expertise.
    data["Industry_Vertical_Expertise"] = (
        normalize_text_series(
            data["Industry_Vertical_Expertise"]
        )
        .fillna("unknown")
        .astype("string")
    )

    # Clean current capacity.
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

    # Clean maximum capacity.
    #
    # Zero and negative capacities become zero, meaning that
    # the representative cannot receive new leads.
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

    return data


# ============================================================
# LEAD SCORING AND SORTING
# ============================================================

def score_leads(
    leads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate lead scores and sort leads from highest priority
    to lowest priority.
    """

    data = leads.copy()

    score = pd.Series(
        0,
        index=data.index,
        dtype="int64",
    )

    # --------------------------------------------------------
    # COMPANY SIZE POINTS
    # --------------------------------------------------------

    company_size_points = np.select(
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

    score += company_size_points

    # --------------------------------------------------------
    # RECENT ACTIVITY POINTS
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

    industry_points = {
        "technology": 15,
        "healthcare": 10,
        "finance": 10,
    }

    score += (
        data["Industry_Vertical"]
        .map(industry_points)
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # VIP OVERRIDE
    # --------------------------------------------------------

    is_vip = (
        data["VIP_Referral"]
        .eq("true")
        .fillna(False)
    )

    data["Score"] = score.mask(
        is_vip,
        100,
    )

    # --------------------------------------------------------
    # PRIORITY CATEGORIES
    # --------------------------------------------------------

    data["Priority"] = np.select(
        [
            data["Score"] >= 85,
            data["Score"] >= 60,
            data["Score"] >= 40,
        ],
        [
            "critical",
            "high",
            "medium",
        ],
        default="low",
    )

    # Create a temporary numeric priority rank.
    data["_Priority_Rank"] = (
        data["Priority"]
        .map(PRIORITY_RANK)
        .fillna(0)
    )

    # Sort by:
    # 1. Priority
    # 2. Score
    # 3. Lead ID
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
    rep: pd.Series,
) -> float:
    """Calculate a representative's capacity utilization."""

    maximum = float(
        rep["Max_Capacity"]
    )

    current = float(
        rep["Current_Capacity"]
    )

    if maximum <= 0:
        return 1.0

    return min(
        current / maximum,
        1.0,
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

    Representatives are ranked by:
        1. Number of matching attributes
        2. Lowest capacity utilization
        3. Highest available capacity
        4. Original representative order
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

            # Skip representatives who have no capacity.
            if (
                max_capacity <= 0
                or current_capacity >= max_capacity
            ):
                continue

            matched_fields = []

            for lead_column, rep_column, label in (
                (
                    "Region",
                    "Region",
                    "region",
                ),
                (
                    "Language",
                    "Language",
                    "language",
                ),
                (
                    "Industry_Vertical",
                    "Industry_Vertical_Expertise",
                    "industry",
                ),
            ):
                lead_value = clean_text(
                    lead[lead_column]
                )

                rep_value = clean_text(
                    rep[rep_column]
                )

                invalid_values = {
                    "",
                    "n/a",
                    "unknown",
                    "none",
                    "nan",
                }

                if (
                    lead_value not in invalid_values
                    and rep_value not in invalid_values
                    and lead_value == rep_value
                ):
                    matched_fields.append(label)

            utilization = capacity_ratio(
                rep
            )

            remaining_capacity = (
                max_capacity
                - current_capacity
            )

            # Larger tuple values win.
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
                int(chosen["Current_Capacity"])
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

    # Keep the final result in priority order.
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

    # Make sure final capacity is numeric.
    roster["Current_Capacity"] = (
        pd.to_numeric(
            roster["Current_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .round()
        .astype("Int64")
    )

    # Calculate final utilization.
    roster["Capacity_Utilization"] = np.where(
        roster["Max_Capacity"] > 0,
        (
            roster["Current_Capacity"]
            / roster["Max_Capacity"]
        ),
        1.0,
    )

    # Calculate remaining capacity.
    roster["Available_Capacity"] = (
        roster["Max_Capacity"]
        - roster["Current_Capacity"]
    ).clip(lower=0)

    # Sort reps with the lowest utilization first.
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
# DOWNLOAD AND DISPLAY HELPERS
# ============================================================

def csv_bytes(
    frame: pd.DataFrame,
) -> bytes:
    """Convert a DataFrame into downloadable CSV bytes."""

    return frame.to_csv(
        index=False,
    ).encode("utf-8-sig")


def company_size_group(
    value: Any,
) -> str:
    """Convert company size into a filter category."""

    if pd.isna(value):
        return "Missing"

    if value < 100:
        return "Under 100"

    if value < 1000:
        return "100–999"

    return "1,000+"


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
# READ FILES
# ============================================================

try:
    raw_leads = clean_column_names(
        pd.read_csv(lead_file)
    )

    raw_reps = clean_column_names(
        pd.read_csv(rep_file)
    )

except (
    UnicodeDecodeError,
    pd.errors.ParserError,
    ValueError,
) as exc:
    st.error(
        f"Could not read one of the CSV files: {exc}"
    )
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_leads = sorted(
    LEAD_REQUIRED
    - set(raw_leads.columns)
)

missing_reps = sorted(
    REP_REQUIRED
    - set(raw_reps.columns)
)

if missing_leads or missing_reps:
    if missing_leads:
        st.error(
            "Leads CSV is missing required columns: "
            + ", ".join(missing_leads)
        )

    if missing_reps:
        st.error(
            "Sales reps CSV is missing required columns: "
            + ", ".join(missing_reps)
        )

    st.stop()


if raw_leads.empty or raw_reps.empty:
    st.error(
        "Both files must contain at least one data row."
    )
    st.stop()


# ============================================================
# CLEAN, SCORE, AND ASSIGN
# ============================================================

leads_clean = normalize_leads(
    raw_leads
)

reps_clean = normalize_reps(
    raw_reps
)

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
    .astype(str)
    .str.strip()
    .ne("")
)

assigned_count = int(
    assigned_mask.sum()
)

unassigned_count = int(
    (~assigned_mask).sum()
)

critical = int(
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
    f"{critical:,}",
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
    st.subheader("Filter and sort leads")

    tracker_col, chart_col = st.columns(
        [2, 1]
    )

    with tracker_col:
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            languages = sorted(
                matched["Language"]
                .fillna("N/A")
                .astype(str)
                .unique()
            )

            language_filter = st.selectbox(
                "Language",
                ["All"] + languages,
            )

        with filter2:
            industries = sorted(
                matched["Industry_Vertical"]
                .fillna("unknown")
                .astype(str)
                .unique()
            )

            industry_filter = st.selectbox(
                "Industry",
                ["All"] + industries,
            )

        with filter3:
            size_groups = (
                matched["Company_Size"]
                .map(company_size_group)
            )

            size_options = [
                "Under 100",
                "100–999",
                "1,000+",
                "Missing",
            ]

            available_sizes = [
                size
                for size in size_options
                if size in size_groups.values
            ]

            size_filter = st.selectbox(
                "Company size",
                ["All"] + available_sizes,
            )

        filtered = matched.copy()

        if language_filter != "All":
            filtered = filtered[
                filtered["Language"]
                .fillna("N/A")
                .astype(str)
                == language_filter
            ]

        if industry_filter != "All":
            filtered = filtered[
                filtered["Industry_Vertical"]
                .fillna("unknown")
                .astype(str)
                == industry_filter
            ]

        if size_filter != "All":
            filtered = filtered[
                filtered["Company_Size"]
                .map(company_size_group)
                == size_filter
            ]

        sort_by = st.selectbox(
            "Sort by",
            [
                "Priority score",
                "Language",
                "Industry",
                "Company size",
            ],
        )

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
            filtered = filtered.copy()

            filtered["_Priority_Rank"] = (
                filtered["Priority"]
                .map(PRIORITY_RANK)
                .fillna(0)
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

        else:
            sort_column = {
                "Language": "Language",
                "Industry": "Industry_Vertical",
            }[sort_by]

            filtered = filtered.sort_values(
                by=[
                    sort_column,
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

        st.caption(
            f"Showing {len(filtered):,} "
            f"of {len(matched):,} leads."
        )

        # Compact lead tracker
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
                filtered["Lead_ID"].astype(str)
                == selected_lead_id
            ]

            if not selected_lead.empty:
                with st.expander(
                    "Selected lead details"
                ):
                    st.json(
                        selected_lead
                        .iloc[0]
                        .to_dict()
                    )

    # --------------------------------------------------------
    # ASSIGNMENT PIE CHART
    # --------------------------------------------------------

    displayed_assigned = int(
        filtered["Assigned_Rep_ID"]
        .fillna("")
        .astype(str)
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
                            "field": "Assignment status",
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

        # ----------------------------------------------------
        # REPRESENTATIVE SUMMARY
        # ----------------------------------------------------

        rep_lead_counts = (
            filtered.loc[
                filtered["Assigned_Rep_ID"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            ]
            .assign(
                _rep_id_key=lambda data: (
                    data["Assigned_Rep_ID"]
                    .astype(str)
                    .str.strip()
                )
            )
            .groupby("_rep_id_key")
            .size()
            .rename("Assigned Leads")
        )

        rep_summary = final_roster[
            [
                "Rep_ID",
                "Name",
            ]
        ].copy()

        rep_summary["_rep_id_key"] = (
            rep_summary["Rep_ID"]
            .astype(str)
            .str.strip()
        )

        rep_summary["Assigned Leads"] = (
            rep_summary["_rep_id_key"]
            .map(rep_lead_counts)
            .fillna(0)
            .astype(int)
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

        st.subheader("Leads by Sales Rep")

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
                == selected_rep_id
            ]

            selected_rep_count = rep_summary.loc[
                rep_summary["Rep_ID"]
                .astype(str)
                == selected_rep_id,
                "Assigned Leads",
            ]

            if not selected_rep.empty:
                rep_details = (
                    selected_rep
                    .iloc[0]
                    .to_dict()
                )

                rep_details[
                    "Assigned Leads (current filters)"
                ] = int(
                    selected_rep_count.iloc[0]
                )

                with st.expander(
                    "Selected rep details"
                ):
                    st.json(rep_details)

    # --------------------------------------------------------
    # DOWNLOAD BUTTONS
    # --------------------------------------------------------

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

    # Guarantee priority order before assigning queue positions.
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
            range(1, len(queue) + 1),
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
        "- Rep assignment is a recommendation: exact region, "
        "language, and industry matches rank first; ties "
        "prefer lower capacity utilization. Each rep is "
        "assigned no more than their positive Max_Capacity."
    )
