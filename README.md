# Lead Matcher

A Streamlit app that accepts two CSV files, cleans their data, scores and prioritizes leads, and recommends sales-rep assignments within each rep's capacity.

## Run locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, then upload both CSV files.

## Required CSV columns

Leads file:

```text
Lead_ID,Company_Name,Region,Language,Industry_Vertical,Company_Size,Recent_Activity,VIP_Referral
```

Sales-reps file:

```text
Rep_ID,Name,Region,Language,Industry_Vertical_Expertise,Seniority_Level,Current_Capacity,Max_Capacity
```

The app cleans the relevant fields in memory and does not write uploaded data to the repository or a local file. It outputs a ranked lead/assignment CSV and a CSV with updated rep capacities.

## Lead scores

- VIP referral: score 100, critical priority.
- Company size: 50 points at 1,000+, 30 at 100–999, 0 below 100, or 20 if missing.
- Recent activity: demo/contact sales 35; webinar/trial 25; whitepaper 15; pricing-page visit 10; blog subscription 5.
- Industry: technology 15; healthcare or finance 10.
- Other priorities: high 60–84, medium 40–59, low 0–39.

Leads are assigned in descending score order. For each lead, an available rep is selected by the number of exact region, language, and industry matches, then lowest capacity utilization. A rep is not assigned above a positive `Max_Capacity`; non-positive maximum capacity is treated as unavailable. If no rep has capacity, the lead is marked unassigned. This assignment rule is an explicit best-fit default because the supplied sorting code scores leads but does not specify a final rep-selection algorithm.

## Deploy

Deploy this repository as a Streamlit app (for example, on Streamlit Community Cloud) with `app.py` as the entry point. The platform installs packages from `requirements.txt`.
