# Lead Matcher

A Streamlit app that accepts two CSV files, cleans their data, scores and prioritizes leads, and recommends sales-rep assignments within each rep's capacity. Team 36's Ignite Challegne.
App Link: https://ignite-challenge-2026.streamlit.app/
## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload a leads CSV and a sales-reps CSV with the required columns shown in the app. Results are split into two tabs: **All leads** provides filtering/sorting, and **Assignment queue** lists unassigned leads in priority-score order with queue positions. The queue can be downloaded as a CSV.
