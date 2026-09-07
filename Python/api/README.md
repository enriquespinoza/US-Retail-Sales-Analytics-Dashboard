# State Retail Sales API

This small FastAPI app downloads three CSVs from the Census MR&T site and exposes them as JSON endpoints.

Endpoints:
- `GET /yoy` — Not Adjusted State Retail Sales YoY CSV
- `GET /standard_errors` — Standard errors CSV
- `GET /coverage` — Coverage measures CSV
- `GET /combined` — Attempts to merge the three datasets

Query params:
- `state` — filter by state (case-insensitive, matches state-like column)
- `year` — filter by year-like column
- `force` — boolean; if true re-downloads the CSV from the source

Run locally:
```
cd Python/api
python -m pip install -r requirements.txt
uvicorn main:app --reload
```
