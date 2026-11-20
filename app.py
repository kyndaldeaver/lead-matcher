st.subheader("Filter and sort leads")

filter1, filter2, filter3, filter4 = st.columns(4)

with filter1:
    languages = sorted(matched["Language"].fillna("N/A").astype(str).unique())
    language_filter = st.selectbox("Language", ["All"] + languages)

with filter2:
    industries = sorted(matched["Industry_Vertical"].fillna("unknown").astype(str).unique())
    industry_filter = st.selectbox("Industry", ["All"] + industries)

with filter3:
    size_groups = matched["Company_Size"].map(company_size_group)
    size_options = ["Under 100", "100–999", "1,000+", "Missing"]
    available_sizes = [size for size in size_options if size in size_groups.values]
    size_filter = st.selectbox("Company size", ["All"] + available_sizes)

with filter4:
    sort_by = st.selectbox(
        "Sort by",
        ["Priority score", "Language", "Industry", "Company size"],
    )

filtered = matched.copy()

if language_filter != "All":
    filtered = filtered[filtered["Language"].fillna("N/A").astype(str) == language_filter]

if industry_filter != "All":
    filtered = filtered[
        filtered["Industry_Vertical"].fillna("unknown").astype(str) == industry_filter
    ]

if size_filter != "All":
    filtered = filtered[
        filtered["Company_Size"].map(company_size_group) == size_filter
    ]

if sort_by == "Company size":
    filtered = filtered.sort_values(
        "Company_Size", ascending=False, na_position="last", kind="stable"
    )
else:
    sort_column = {
        "Priority score": "Score",
        "Language": "Language",
        "Industry": "Industry_Vertical",
    }[sort_by]
    filtered = filtered.sort_values(
        sort_column, ascending=True, na_position="last", kind="stable"
    )

st.caption(f"Showing {len(filtered):,} of {len(matched):,} leads.")
st.dataframe(filtered, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered leads and assignments",
    data=csv_bytes(filtered),
    file_name="ranked_leads_with_assignments.csv",
    mime="text/csv",
)

st.download_button(
    "Download updated rep capacities",
    data=csv_bytes(final_roster),
    file_name="sales_reps_updated_capacity.csv",
    mime="text/csv",
)
