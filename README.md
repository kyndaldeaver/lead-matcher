# Lead Matcher

A Streamlit app that accepts two CSV files, cleans their data, scores and prioritizes leads, and recommends sales-rep assignments within one rep capacity.

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

## Filter and sort

The results table is controlled by dropdown filters for language, industry, and company-size group. To view all leads, select **All** in each filter. The sort dropdown changes the order of the displayed table by priority score, language, industry, or company size.

Only the displayed table is offered as a CSV download. This avoids a separate, redundant all-leads view.
