from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Lead Matcher",
    layout="wide",
)


# ============================================================
# REQUIRED COLUMNS
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
# HEADER CLEANING
# ============================================================

HEADER_ALIASES = {
    "leadid": "Lead_ID",
    "companyname": "Company_Name",
    "region": "Region",
    "language": "Language",
    "industryvertical": "Industry_Vertical",
    "companysize": "Company_Size",
    "recentactivity": "Recent_Activity",
    "vipreferral": "VIP_Referral",

    "repid": "Rep_ID",
    "name": "Name",
    "industryverticalexpertise":
        "Industry_Vertical_Expertise",
    "currentcapacity": "Current_Capacity",
    "maxcapacity": "Max_Capacity",
}


def header_key(value: Any) -> str:
    """
    Convert a column header into a simple lookup key.

    Examples:
        "Lead ID"       -> "leadid"
        "Lead_ID"       -> "leadid"
        " Lead-ID "     -> "leadid"
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
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


# ============================================================
# GENERAL NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text_series(
    series: pd.Series,
) -> pd.Series:
    """
    Clean an entire text column.

    This removes outside spaces, converts values to lowercase,
    and replaces repeated spaces with one space.
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
    Create dictionary lookup keys by removing spaces and
    punctuation.

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
    """Clean one text value."""

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def clean_region(value: Any) -> str:
    """Clean one region value."""

    raw = clean_text(value)

    if not raw:
        return "N/A"

    key = re.sub(
        r"[^a-z0-9]+",
        "",
        raw,
    )

    return REGION_ALIASES.get(
        key,
        raw,
    )


def clean_language(value: Any) -> str:
    """Clean one language value."""

    raw = clean_text(value)

    if not raw:
        return "N/A"

    key = re.sub(
        r"[^a-z0-9]+",
        "",
        raw,
    )

    return LANGUAGE_ALIASES.get(
        key,
        "N/A",
    )


def clean_activity(value: Any) -> str:
    """Clean one recent-activity value."""

    raw = clean_text(value)

    if not raw:
        return "Unknown"

    key = re.sub(
        r"[^a-z0-9]+",
        "",
        raw,
    )

    return ACTIVITY_ALIASES.get(
        key,
        raw.title(),
    )


def company_size(value: Any) -> float | None:
    """Clean one company-size value."""

    if pd.isna(value):
        return None

    raw = (
        clean_text(value)
        .replace(",", "")
        .replace(" ", "")
    )

    if raw in {
        "",
        "n/a",
        "na",
        "none",
        "nan",
        "unknown",
    }:
        return None

    match = re.search(
        r"[0-9]*\.?[0-9]+",
        raw,
    )

    if match is None:
        return None

    number = float(match.group())

    if raw.endswith("k"):
        number *= 1000

    return number


def clean_company_size_series(
    series: pd.Series,
) -> pd.Series:
    """
    Clean the entire Company_Size column.

    Examples:
        "500"   -> 500
        "2.5k"  -> 2500
        "1,200" -> 1200
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

    return (
        numbers
        .mul(multiplier)
        .round()
        .astype("Float64")
    )


# ============================================================
# LEAD CLEANING
# ============================================================

def normalize_leads(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and normalize all lead columns."""

    data = frame.copy()

    data["Lead_ID"] = (
        data["Lead_ID"]
        .astype("string")
        .str.strip()
    )

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

    # Clean region keys and map them to standard values.
    raw_regions = normalize_text_series(
        data["Region"]
    )

    region_keys = normalize_key_series(
        data["Region"]
    )

    data["Region"] = (
        region_keys
        .map(REGION_ALIASES)
        .fillna(raw_regions)
        .fillna("N/A")
        .astype("string")
    )

    # Clean language keys and map them to standard values.
    language_keys = normalize_key_series(
        data["Language"]
    )

    data["Language"] = (
        language_keys
        .map(LANGUAGE_ALIASES)
        .fillna("N/A")
        .astype("string")
    )

    data["Industry_Vertical"] = (
        normalize_text_series(
            data["Industry_Vertical"]
        )
        .fillna("unknown")
    )

    data["Company_Size"] = (
        clean_company_size_series(
            data["Company_Size"]
        )
    )

    activity_keys = normalize_key_series(
        data["Recent_Activity"]
    )

    raw_activity = normalize_text_series(
        data["Recent_Activity"]
    )

    data["Recent_Activity"] = (
        activity_keys
        .map(ACTIVITY_ALIASES)
        .fillna(
            raw_activity.str.title()
        )
        .fillna("Unknown")
        .astype("string")
    )

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
# REPRESENTATIVE CLEANING
# ============================================================

def normalize_reps(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and normalize all representative columns."""

    data = frame.copy()

    data["Rep_ID"] = (
        data["Rep_ID"]
        .astype("string")
        .str.strip()
    )

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

    raw_regions = normalize_text_series(
        data["Region"]
    )

    region_keys = normalize_key_series(
        data["Region"]
    )

    data["Region"] = (
        region_keys
        .map(REGION_ALIASES)
        .fillna(raw_regions)
        .fillna("N/A")
        .astype("string")
    )

    language_keys = normalize_key_series(
        data["Language"]
    )

    data["Language"] = (
        language_keys
        .map(LANGUAGE_ALIASES)
        .fillna("N/A")
        .astype("string")
    )

    data["Industry_Vertical_Expertise"] = (
        normalize_text_series(
            data[
                "Industry_Vertical_Expertise"
            ]
        )
        .fillna("unknown")
    )

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

    score += (
        data["Recent_Activity"]
        .map(ACTIVITY_POINTS)
        .fillna(0)
        .astype("int64")
    )

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

    is_vip = (
        data["VIP_Referral"]
        .eq("true")
        .fillna(False)
    )

    data["Score"] = score.mask(
        is_vip,
        100,
    )

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

    priority_rank = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    data["_Priority_Rank"] = (
        data["Priority"]
        .map(priority_rank)
        .fillna(0)
    )

    # Highest priority first. Within the same priority,
    # highest score appears first.
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
# CAPACITY CALCULATION
# ============================================================

def capacity_ratio(rep: pd.Series) -> float:
    """Calculate how much of a rep's capacity is being used."""

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
# MATCHING AND ASSIGNMENT
# ============================================================

def match_and_assign(
    leads: pd.DataFrame,
    reps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign highest-priority leads first.

    Representatives are ranked using:
        1. Number of matching attributes
        2. Lowest capacity utilization
        3. Highest remaining capacity
        4. Original representative order
    """

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
                float(
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

    # Preserve the queue order:
    # critical, high, medium, low; then highest score.
    priority_rank = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    result["_Priority_Rank"] = (
        result["Priority"]
        .map(priority_rank)
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

    roster["Current_Capacity"] = (
        pd.to_numeric(
            roster["Current_Capacity"],
            errors="coerce",
        )
        .fillna(0)
        .round()
        .astype("Int64")
    )

    roster["Capacity_Utilization"] = np.where(
        roster["Max_Capacity"] > 0,
        (
            roster["Current_Capacity"]
            / roster["Max_Capacity"]
        ),
        1.0,
    )

    roster["Available_Capacity"] = (
        roster["Max_Capacity"]
        - roster["Current_Capacity"]
    ).clip(lower=0)

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
    """Convert a company size into a filter category."""

    if pd.isna(value):
        return "Missing"

    if value < 100:
        return "Under 100"

    if value < 1000:
        return "100–999"

    return "1,000+"
