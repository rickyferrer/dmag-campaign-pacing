# D Magazine Campaign Pacing MVP (Retool + Postgres)

This project automates ad campaign pacing from a CSV export and sends risk alerts by email.

## What this does

1. Ingests your pacing CSV export.
2. Calculates pacing health and projected final delivery.
3. Stores campaign snapshots in Postgres.
4. Sends alert emails for at-risk campaigns to `ricky.ferrer@dmagazine.com`.
5. Provides SQL queries for a Retool dashboard.

## Quick start

1. Copy env template:

```bash
cp .env.example .env
```

2. Update `.env` values (Postgres and SMTP).

3. Initialize database:

```bash
psql "$DATABASE_URL" -f sql/schema.sql
```

4. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

5. Run ingestion + alerts:

```bash
python3 src/main.py --csv /absolute/path/to/pacing_export.csv
```

Optional:

```bash
python3 src/main.py --csv /absolute/path/to/pacing_export.csv --dry-run
```

`--dry-run` calculates pacing and prints risk output without writing to DB or sending email.

## Run from Google Ad Manager (no CSV export)

1. Configure GAM settings in `.env`:

- `GAM_NETWORK_CODE`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GAM_FIELD_MAP` (optional if report column order matches defaults)

2. Run:

```bash
python3 src/main.py --source gam --gam-report-id YOUR_REPORT_ID
```

Dry run:

```bash
python3 src/main.py --source gam --gam-report-id YOUR_REPORT_ID --dry-run
```

## CSV column expectations

Default headers expected in the CSV:

- `campaign_id`
- `campaign_name`
- `advertiser`
- `start_date`
- `end_date`
- `goal_impressions`
- `delivered_impressions`

If your headers differ, set `CSV_COLUMN_MAP` in `.env` (JSON object).

This importer also supports your current sheet format with repeated monthly blocks like:

- `CLIENT, REP, IO, ... , STATUS, START, END, ... , IMPRESSIONS, GOAL, ...`

It also supports Google Ad Manager CSV exports with metadata rows and headers like:

- `Line item ID, Ad unit status, Line item start date, Line item end date, Line item, Order, Line item primary goal units (absolute), Ad server impressions`

For accurate pacing from GAM CSV, use a cumulative date range (for example line item lifetime or flight-to-date).  
If the report date range is only one day, delivered impressions will be too low and most campaigns will appear high risk.

## Risk logic

- `expected_to_date = goal * (days_elapsed / total_days)`
- `pacing_pct = delivered / expected_to_date`
- `projected_final = delivered + (avg_daily_rate * days_remaining)`
- `high`: projected_final < 90% of goal
- `medium`: projected_final < 95% of goal
- `watch`: pacing_pct < 95%

## Retool dashboard setup

Use queries in `sql/retool_queries.sql` against your Postgres resource:

1. `portfolio_summary`
2. `campaign_table`
3. `campaign_trend`
4. `risk_alert_feed`

Build a table + charts and add filters for advertiser/date/risk level.

Full setup details: `docs/SETUP_RETOOL.md`

## Streamlit dashboard (quick visual)

You can run a local visual dashboard directly from this project:

```bash
streamlit run dashboard/app.py
```

It shows:

- Portfolio KPIs (high/medium/watch/on-track)
- Risk distribution chart
- Top projected gap-to-goal campaigns
- Risk trend over time
- Filterable campaign detail table

## Suggested automation schedule

- Run every weekday at 7:00 AM CT.
- Optionally run again midday.
- Feed output to Retool and send risk alerts by email.

## Fully automatic updates (GitHub Actions + Streamlit Cloud)

This repo includes `.github/workflows/gam-pacing-refresh.yml` to run ingestion automatically from GAM and write snapshots to Postgres.

### 1) Push this repo to GitHub

```bash
cd "/Users/rickyferrer/Working Files/Codex Project"
git add .
git commit -m "Add automated GAM refresh workflow"
git push
```

### 2) Add GitHub Actions secrets

In GitHub: `Repo -> Settings -> Secrets and variables -> Actions -> New repository secret`

Add:

- `DATABASE_URL`
- `GAM_NETWORK_CODE`
- `GAM_REPORT_ID`
- `GAM_SERVICE_ACCOUNT_JSON` (paste full JSON contents, not a file path)
- `GAM_FIELD_MAP` (optional; only if you need custom column mapping)
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `ALERT_TO_EMAIL`

### 3) Test once manually

In GitHub: `Repo -> Actions -> GAM Pacing Refresh -> Run workflow`

Check logs for:

- `Schema ready.`
- `Snapshot written to DB and alert email sent.`

### 4) Deploy dashboard for team access

Use `docs/DEPLOY_STREAMLIT_CLOUD.md`.

In Streamlit secrets, set:

```toml
DATABASE_URL = "postgresql://..."
```

After this, your dashboard updates automatically as each scheduled GitHub run writes new snapshots.

## Data quality note

Your current sample CSV (`Digital Inventory Placement 2026 - ROS.csv`) does not include delivered impressions.  
Without a delivered-impressions field from GAM, pacing risk cannot be measured accurately.
