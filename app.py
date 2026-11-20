from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Lead Matcher", layout="wide")

LEAD_REQUIRED = {
    "Lead_ID", "Company_Name", "Region", "Language", "Industry_Vertical",
    "Company_Size", "Recent_Activity", "VIP_Referral",
}
REP_REQUIRED = {
    "Rep_ID", "Name", "Region", "Language", "Industry_Vertical_Expertise",
    "Current_Capacity", "Max_Capacity",
}

REGION_ALIASES = {
    "asa-pacific": "asia pacific", "asia-pacific": "asia pacific",
    "asia.pacific": "asia pacific", "asiapacific": "asia pacific",
    "apac": "asia pacific", "europ": "europe", "europe": "europe",
    "eur": "europe", "asiapacificeurope": "asia pacific and europe",
    "northamerica": "north america", "nortamer": "north america",
    "nortamerica": "north america", "northamer": "north america",
    "northamericaasia-pacific": "north america and asia pacific",
    "northamericaasiapacific": "north america and asia pacific",
}
LANGUAGE_ALIASES = {
    "es": "spanish", "spanish": "spanish", "english": "english", "en": "english",
    "german": "german", "de": "german",
}
ACTIVITY_ALIASES = {
    "n/a": "N/A", "unknown": "Unknown", "subscribed": "Subscribed To Blog",
    "downloadedwhitepaper": "Downloaded Whitepaper",
    "requestedlivedemo": "Requested Live Demo",
    "openedmarketingemail": "Opened Marketing Email", "contactsales": "Contact Sales",
    "visitedpricingpage": "Visited Pricing Page", "startedfreetrial": "Started Free Trial",
    "attendedwebinar": "Attended Webinar", "subscribedtoblog": "Subscribed To Blog",
    "": "Unknown", "nan": "Unknown", "none": "Unknown",
}
ACTIVITY_POINTS = {
    "Requested Live Demo": 35, "Contact Sales": 35,
    "Attended Webinar": 25, "Started Free Trial": 25,
    "Downloaded Whitepaper": 15, "Visited Pricing Page": 10,
    "Subscribed To Blog": 5,
}


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def clean_region(value: Any) -> str:
    raw = clean_text(value)
    key = "".join(raw.split())
    if not key:
        return "N/A"
    return REGION_ALIASES.get(key, raw)


def clean_language(value: Any) -> str:
    raw = clean_text(value)
    return LANGUAGE_ALIASES.get(raw, "N/A") if raw else "N/A"


def clean_activity(value: Any) -> str:
    raw = clean_text(value)
    key = "".join(raw.split())
    if key in ACTIVITY_ALIASES:
        return ACTIVITY_ALIASES[key]
    return raw.title() if raw else "Unknown"


def company_size(value: Any) -> float | None:
    if pd.isna(value):
        return None
    raw = clean_text(value).replace(",", "")
    if raw in {"", "n/a", "na", "none", "nan", "unknown"}:
        return None
    try:
        multiplier = 1000 if raw.endswith("k") else 1
        return float(raw[:-1] if multiplier == 1000 else raw) * multiplier
    except ValueError:
        return None


def normalize_leads(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()*

