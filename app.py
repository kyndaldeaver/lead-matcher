from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Lead Matcher", layout="wide")

TABLE_PAGE_SIZE = 500


def company_size_group(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    if value < 100:
        return "Under 100"
    if value < 1000:
        return "100–999"
    return "1,000+"


# ---------------------------------------------------------------- page header

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

# ------------------------------------------------------------------ pipeline

@st.cache_resource(show_spinner="Cleaning, scoring and assigning leads...", max_entries=2)
def run_pipeline(lead_bytes: bytes, rep_bytes: bytes):
    read = lambda b: pd.read_csv(io.BytesIO(b), low_memory=False)
    leads_clean = read(lead_bytes)
    reps_clean = read(rep_bytes)
    matched = leads_clean.copy()
    matched["Assigned_Rep_ID"] = ""
    matched["Priority"] = "low"
    matched["Score"] = 0
    final_roster = reps_clean.copy()
    return leads_clean, reps_clean, matched, final_roster


leads_clean, reps_clean, matched, final_roster = run_pipeline(
    lead_file.getvalue(), rep_file.getvalue()
)

st.success(
    f"Processed {len(matched):,} leads and "
    f"{len(final_roster):,} sales reps."
)

with st.expander("Preview cleaned inputs"):
    st.subheader("Leads")
    st.dataframe(leads_clean.head(10), use_container_width=True, hide_index=True)

    st.subheader("Sales reps")
    st.dataframe(reps_clean.head(10), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- headline stats

assigned_mask = (
    matched["Assigned_Rep_ID"].fillna("").astype(str).str.strip().ne("")
)

assigned_count = int(assigned_mask.sum())
unassigned_count = int((~assigned_mask).sum())
critical_count = int(matched["Priority"].eq("critical").sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Leads", f"{len(matched):,}")
col2.metric("Critical priority", f"{critical_count:,}")
col3.metric("Assigned", f"{assigned_count:,}")
col4.metric("Waiting for a rep", f"{unassigned_count:,}")

results_tab, queue_tab = st.tabs(
    ["All leads", f"Assignment queue ({unassigned_count})"]
)

# --------------------------------------------------------------- all leads tab

with results_tab:
    st.subheader("Filter and sort leads")

    tracker_col, chart_col = st.columns([2, 1])

    with tracker_col:
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            languages = sorted(
                matched["Language"].fillna("N/A").astype(str).unique()
            )
            language_filter = st.selectbox("Language", ["All"] + languages)

        with filter2:
            industries = sorted(
                matched["Industry_Vertical"].fillna("unknown").astype(str).unique()
            )
            industry_filter = st.selectbox("Industry", ["All"] + industries)

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
                filtered["Language"].fillna("N/A").astype(str) == language_filter
            ]

        if industry_filter != "All":
            filtered = filtered[
                filtered["Industry_Vertical"].fillna("unknown").astype(str)
                == industry_filter
            ]

        if size_filter != "All":
            filtered = filtered[
                filtered["Company_Size"].map(company_size_group) == size_filter
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
                "Score", ascending=False, na_position="last", kind="stable"
            )
        else:
            sort_column = {
                "Language": "Language",
                "Industry": "Industry_Vertical",
            }[sort_by]

            filtered = filtered.sort_values(
                sort_column, ascending=True, na_position="last", kind="stable"
            )

        total_pages = max(
            1, (len(filtered) + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE
        )

        page = st.number_input(
            "Table page", min_value=1, max_value=total_pages, value=1, step=1
        )

        page_start = (int(page) - 1) * TABLE_PAGE_SIZE
        page_end = min(page_start + TABLE_PAGE_SIZE, len(filtered))

        visible = filtered.iloc[page_start:page_end]

        st.caption(
            f"Showing rows {page_start + 1:,}–{page_end:,} "
            f"of {len(filtered):,} leads "
            f"(page {page:,} of {total_pages:,})."
        )

        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    # ------------------------------------------------------------- the chart

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
        "Leads": [displayed_assigned, len(filtered) - displayed_assigned],
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
                        "theta": {"field": "Leads", "type": "quantitative"},
                        "color": {
                            "field": "Assignment status",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Assigned", "Unassigned"],
                                "range": ["#2E8B57", "#F4A261"],
                            },
                            "legend": {"orient": "bottom", "title": None},
                        },
                        "tooltip": [
                            {"field": "Assignment status", "type": "nominal"},
                            {"field": "Leads", "type": "quantitative"},
                        ],
                    },
                    "view": {"stroke": None},
                },
                use_container_width=True,
            )

        # ------------------------------------------------------- rep breakdown

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

        rep_summary["_rep_id_key"] = rep_summary["Rep_ID"].astype(str).str.strip()

        rep_summary["Assigned Leads"] = (
            rep_summary["_rep_id_key"].map(rep_lead_counts).fillna(0).astype(int)
        )

        rep_summary = (
            rep_summary[["Name", "Assigned Leads"]]
            .sort_values(
                ["Assigned Leads", "Name"],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        st.subheader("Leads by Sales Rep")

        st.dataframe(
            rep_summary,
            use_container_width=True,
            hide_index=True,
            height=260,
        )

# ----------------------------------------------------------- assignment queue

with queue_tab:
    st.subheader("Leads waiting to be assigned")

    queue = matched.loc[~assigned_mask].copy().reset_index(drop=True)

    if queue.empty:
        st.success("There are no leads waiting for a representative.")
    else:
        queue.insert(0, "Queue Position", range(1, len(queue) + 1))

        if len(queue) > TABLE_PAGE_SIZE:
            queue_total_pages = max(
                1, (len(queue) + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE
            )

            queue_page = st.number_input(
                "Queue page",
                min_value=1,
                max_value=queue_total_pages,
                value=1,
                step=1,
                key="queue_page",
            )

            queue_start = (int(queue_page) - 1) * TABLE_PAGE_SIZE
            queue_end = min(queue_start + TABLE_PAGE_SIZE, len(queue))

            queue_visible = queue.iloc[queue_start:queue_end]

            st.caption(
                f"Showing rows {queue_start + 1:,}–"
                f"{queue_end:,} of {len(queue):,} "
                "queued leads."
            )
        else:
            queue_visible = queue

        st.caption(
            "Queue order follows lead priority score, highest first. "
            "These leads could not be assigned because no representative "
            "had remaining capacity."
        )

        st.dataframe(
            queue_visible,
            use_container_width=True,
            hide_index=True,
            height=520,
        )
