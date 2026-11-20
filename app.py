from __future__ import annotations

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
    data = frame.copy()
    data["Region"] = data["Region"].map(clean_region)
    data["Language"] = data["Language"].map(clean_language)
    data["Industry_Vertical"] = data["Industry_Vertical"].map(
        lambda value: clean_text(value) or "unknown"
    )
    data["Company_Size"] = data["Company_Size"].map(company_size)
    data["Recent_Activity"] = data["Recent_Activity"].map(clean_activity)
    data["VIP_Referral"] = data["VIP_Referral"].map(
        lambda value: clean_text(value)
        if clean_text(value) in {"true", "false", "unknown"}
        else "unknown"
    )
    return data


def normalize_reps(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Name"] = data["Name"].map(
        lambda value: re.split(r"\s*\(", str(value), maxsplit=1)[0].strip()
        if pd.notna(value)
        else ""
    )
    data["Region"] = data["Region"].map(clean_region)
    data["Language"] = data["Language"].map(clean_language)
    data["Industry_Vertical_Expertise"] = data["Industry_Vertical_Expertise"].map(
        lambda value: clean_text(value) or "unknown"
    )
    for column in ("Current_Capacity", "Max_Capacity"):
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(0).clip(lower=0)
    return data


def score_leads(leads: pd.DataFrame) -> pd.DataFrame:
    data = leads.copy()

    def score_one(lead: pd.Series) -> int:
        if lead["VIP_Referral"] == "true":
            return 100
        score = 20 if pd.isna(lead["Company_Size"]) else (
            50 if lead["Company_Size"] >= 1000
            else 30 if lead["Company_Size"] >= 100
            else 0
        )
        score += ACTIVITY_POINTS.get(lead["Recent_Activity"], 0)
        vertical = lead["Industry_Vertical"]
        score += (
            15 if vertical == "technology"
            else 10 if vertical in {"healthcare", "finance"}
            else 0
        )
        return score

    data["Score"] = data.apply(score_one, axis=1)
    data["Priority"] = pd.cut(
        data["Score"],
        bins=[-1, 39, 59, 84, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype("string")
    data.loc[data["VIP_Referral"] == "true", "Priority"] = "critical"
    return data.sort_values(
        "Score", ascending=False, kind="stable"
    ).reset_index(drop=True)


def capacity_ratio(rep: pd.Series) -> float:
    maximum = float(rep["Max_Capacity"])
    current = float(rep["Current_Capacity"])
    if maximum <= 0:
        return 1.0
    return min(current / maximum, 1.0)


def match_and_assign(
    leads: pd.DataFrame, reps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = score_leads(leads)
    roster = reps.copy().reset_index(drop=True)
    assignments: list[dict[str, Any]] = []

    for _, lead in ranked.iterrows():
        candidates: list[tuple[tuple[float, float, float], int, list[str]]] = []
        for rep_index, rep in roster.iterrows():
            max_capacity = float(rep["Max_Capacity"])
            current_capacity = float(rep["Current_Capacity"])
            if max_capacity <= 0 or current_capacity >= max_capacity:
                continue

            matched: list[str] = []
            for lead_col, rep_col, label in (
                ("Region", "Region", "region"),
                ("Language", "Language", "language"),
                ("Industry_Vertical", "Industry_Vertical_Expertise", "industry"),
            ):
                lead_value = str(lead[lead_col]).lower()
                rep_value = str(rep[rep_col]).lower()
                if (
                    lead_value not in {"n/a", "unknown", ""}
                    and rep_value not in {"n/a", "unknown", ""}
                    and lead_value == rep_value
                ):
                    matched.append(label)

            candidates.append(
                (
                    (
                        float(len(matched)),
                        -capacity_ratio(rep),
                        -current_capacity,
                    ),
                    rep_index,
                    matched,
                )
            )

        if candidates:
            _, chosen_index, matched_fields = max(
                candidates, key=lambda item: (item[0], -item[1])
            )
            chosen = roster.loc[chosen_index]
            roster.at[chosen_index, "Current_Capacity"] = (
                float(chosen["Current_Capacity"]) + 1
            )
            assignments.append({
                "Assigned_Rep_ID": chosen["Rep_ID"],
                "Assigned_Rep": chosen["Name"],
                "Rep_Match": (
                    ", ".join(matched_fields)
                    if matched_fields
                    else "No exact attribute match"
                ),
                "Rep_Capacity_Status": "Assigned within capacity",
            })
        else:
            assignments.append({
                "Assigned_Rep_ID": "",
                "Assigned_Rep": "",
                "Rep_Match": "",
                "Rep_Capacity_Status": "No rep capacity available",
            })

    result = pd.concat([ranked, pd.DataFrame(assignments)], axis=1)
    return result, roster


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def company_size_group(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    if value < 100:
        return "Under 100"
    if value < 1000:
        return "100–999"
    return "1,000+"


st.title("Lead Matcher")
st.write(
    "Upload lead and sales-representative CSV files to clean, prioritize, "
    "and assign leads."
)

with st.sidebar:
    st.header("Upload files")
    lead_file = st.file_uploader("Leads CSV", type=["csv"], key="leads")
    rep_file = st.file_uploader("Sales reps CSV", type=["csv"], key="reps")
    st.caption(
        "Upload separate CSV files with header rows. "
        "See README for required columns."
    )

if not lead_file or not rep_file:
    st.info("Upload both CSV files to start matching.")
    st.stop()

try:
    raw_leads = pd.read_csv(lead_file)
    raw_reps = pd.read_csv(rep_file)
except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
    st.error(f"Could not read one of the CSV files: {exc}")
    st.stop()

missing_leads = sorted(LEAD_REQUIRED - set(raw_leads.columns))
missing_reps = sorted(REP_REQUIRED - set(raw_reps.columns))
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
    st.error("Both files must contain at least one data row.")
    st.stop()

leads_clean = normalize_leads(raw_leads)
reps_clean = normalize_reps(raw_reps)
matched, final_roster = match_and_assign(leads_clean, reps_clean)

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
assigned_count = int(assigned_mask.sum())
unassigned_count = int((~assigned_mask).sum())
critical = matched["Priority"].eq("critical").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Leads", f"{len(matched):,}")
col2.metric("Critical priority", f"{critical:,}")
col3.metric("Assigned", f"{assigned_count:,}")
col4.metric("Waiting for a rep", f"{unassigned_count:,}")

results_tab, queue_tab = st.tabs(
    ["All leads", f"Assignment queue ({unassigned_count})"]
)

with results_tab:
    st.subheader("Filter and sort leads")
    tracker_col, chart_col = st.columns([2, 1])

    with tracker_col:
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            languages = sorted(
                matched["Language"].fillna("N/A").astype(str).unique()
            )
            language_filter = st.selectbox(
                "Language", ["All"] + languages
            )

        with filter2:
            industries = sorted(
                matched["Industry_Vertical"]
                .fillna("unknown")
                .astype(str)
                .unique()
            )
            industry_filter = st.selectbox(
                "Industry", ["All"] + industries
            )

        with filter3:
            size_groups = matched["Company_Size"].map(company_size_group)
            size_options = ["Under 100", "100–999", "1,000+", "Missing"]
            available_sizes = [
                size for size in size_options if size in size_groups.values
            ]
            size_filter = st.selectbox(
                "Company size", ["All"] + available_sizes
            )

        filtered = matched.copy()
        if language_filter != "All":
            filtered = filtered[
                filtered["Language"].fillna("N/A").astype(str)
                == language_filter
            ]
        if industry_filter != "All":
            filtered = filtered[
                filtered["Industry_Vertical"].fillna("unknown").astype(str)
                == industry_filter
            ]
        if size_filter != "All":
            filtered = filtered[
                filtered["Company_Size"].map(company_size_group)
                == size_filter
            ]

        sort_by = st.selectbox(
            "Sort by",
            ["Priority score", "Language", "Industry", "Company size"],
        )
        if sort_by == "Company size":
            filtered = filtered.sort_values(
                "Company_Size",
                ascending=False,
                na_position="last",
                kind="stable",
            )
        elif sort_by == "Priority score":
            filtered = filtered.sort_values(
                "Score",
                ascending=False,
                na_position="last",
                kind="stable",
            )
        else:
            sort_column = {
                "Language": "Language",
                "Industry": "Industry_Vertical",
            }[sort_by]
            filtered = filtered.sort_values(
                sort_column,
                ascending=True,
                na_position="last",
                kind="stable",
            )

        st.caption(
            f"Showing {len(filtered):,} of {len(matched):,} leads."
        )

        # Keep the tracker compact and show complete lead information
        # after a lead is selected.
        lead_preview = filtered[["Lead_ID", "Company_Name"]].copy()
        st.dataframe(
            lead_preview,
            use_container_width=True,
            hide_index=True,
        )
        if not lead_preview.empty:
            lead_options = lead_preview["Lead_ID"].astype(str).tolist()
            selected_lead_id = st.selectbox(
                "Select a lead to see its full details",
                lead_options,
                key="lead_details_selector",
            )
            selected_lead = filtered.loc[
                filtered["Lead_ID"].astype(str) == selected_lead_id
            ]
            if not selected_lead.empty:
                with st.expander("Selected lead details"):
                    st.json(selected_lead.iloc[0].to_dict())

    displayed_assigned = int(
        filtered["Assigned_Rep_ID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    assignment_chart = pd.DataFrame({
        "Assignment status": ["Assigned", "Unassigned"],
        "Leads": [
            displayed_assigned,
            len(filtered) - displayed_assigned,
        ],
    })

    with chart_col:
        st.markdown(
            "<h3 style='text-align: center;'>"
            "Assigned vs. Unassigned Leads</h3>",
            unsafe_allow_html=True,
        )
        chart_left, chart_center, chart_right = st.columns([1, 4, 1])

        with chart_center:
            st.vega_lite_chart(
                assignment_chart,
                {
                    "width": 260,
                    "height": 300,
                    "mark": {"type": "arc", "outerRadius": 105},
                    "encoding": {
                        "theta": {
                            "field": "Leads",
                            "type": "quantitative",
                        },
                        "color": {
                            "field": "Assignment status",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Assigned", "Unassigned"],
                                "range": ["#2E8B57", "#F4A261"],
                            },
                            "legend": {
                                "orient": "bottom",
                                "title": None,
                            },
                        },
                        "tooltip": [
                            {
                                "field": "Assignment status",
                                "type": "nominal",
                            },
                            {
                                "field": "Leads",
                                "type": "quantitative",
                            },
                        ],
                    },
                    "view": {"stroke": None},
                },
                use_container_width=True,
            )

        # Count assigned leads in the filtered results and include every rep,
        # including reps with no displayed leads.
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
                    data["Assigned_Rep_ID"].astype(str).str.strip()
                )
            )
            .groupby("_rep_id_key")
            .size()
            .rename("Assigned Leads")
        )

        rep_summary = final_roster[["Rep_ID", "Name"]].copy()
        rep_summary["_rep_id_key"] = (
            rep_summary["Rep_ID"].astype(str).str.strip()
        )
        rep_summary["Assigned Leads"] = (
            rep_summary["_rep_id_key"]
            .map(rep_lead_counts)
            .fillna(0)
            .astype(int)
        )
        rep_summary = (
            rep_summary[["Rep_ID", "Name", "Assigned Leads"]]
            .sort_values(
                ["Assigned Leads", "Name"],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        st.subheader("Leads by Sales Rep")
        st.dataframe(
            rep_summary[["Rep_ID", "Name"]],
            use_container_width=True,
            hide_index=True,
        )
        if not rep_summary.empty:
            rep_options = {
                f"{row.Name} (ID: {row.Rep_ID})": str(row.Rep_ID)
                for row in rep_summary.itertuples(index=False)
            }
            selected_rep_label = st.selectbox(
                "Select a rep to see lead count and full details",
                list(rep_options),
                key="rep_details_selector",
            )
            selected_rep_id = rep_options[selected_rep_label]
            selected_rep = final_roster.loc[
                final_roster["Rep_ID"].astype(str) == selected_rep_id
            ]
            selected_rep_count = rep_summary.loc[
                rep_summary["Rep_ID"].astype(str) == selected_rep_id,
                "Assigned Leads",
            ]
            if not selected_rep.empty:
                rep_details = selected_rep.iloc[0].to_dict()
                rep_details["Assigned Leads (current filters)"] = int(
                    selected_rep_count.iloc[0]
                )
                with st.expander("Selected rep details"):
                    st.json(rep_details)

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

with queue_tab:
    st.subheader("Leads waiting to be assigned")
    queue = matched.loc[~assigned_mask].copy().reset_index(drop=True)

    if queue.empty:
        st.success("There are no leads waiting for a representative.")
    else:
        queue.insert(0, "Queue Position", range(1, len(queue) + 1))
        st.caption(
            "Queue order follows lead priority score, highest first. "
            "These leads could not be assigned because no representative "
            "had remaining capacity."
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

with st.expander("Scoring and assignment rules"):
    st.markdown(
        "- VIP referrals score **100** and are always **critical**.\n"
        "- Company size: **50** points for 1,000+, **30** for 100–999, "
        "**0** below 100, or **20** when missing.\n"
        "- Recent activity: demo/contact sales **35**, webinar/trial **25**, "
        "whitepaper **15**, pricing page **10**, blog subscription **5**.\n"
        "- Industry: technology **15**, healthcare/finance **10**.\n"
        "- Other priority bands: high **60–84**, medium **40–59**, "
        "low **0–39**.\n"
        "- Rep assignment is a recommendation: exact region, language, and "
        "industry matches rank first; ties prefer lower capacity utilization. "
        "Each rep is assigned no more than their positive Max_Capacity."
    )
