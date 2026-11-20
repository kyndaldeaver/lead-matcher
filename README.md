# Lead Matcher

A Streamlit app that accepts two CSV files, cleans their data, scores and prioritizes leads, and recommends sales-rep assignments within each rep's capacity.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Required CSV columns

Leads file:

```text
Lead_ID,Company_Name,Region,Language,Industry_Vertical,Company_Size,Recent_Activity,VIP_Referral
```

Sales-reps file:

```text
Rep_ID,Name,Region,Language,Industry_Vertical_Expertise,Seniority_Level,Current_Capacity,Max_Capacity
```

## Filter and sort controls

The results page provides dropdown filters for language, industry, and company-size group (under 100, 100–999, 1,000+, or missing). The sort dropdown orders the filtered table by priority score, language, industry, or company size.

The filtered table is shown below the dropdowns. An expandable **View all leads** section shows every processed lead, regardless of filters.
