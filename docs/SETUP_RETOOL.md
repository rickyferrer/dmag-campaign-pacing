# Retool Setup (D Magazine Pacing)

## 1) Create Postgres resource in Retool

1. In Retool, add a new `PostgreSQL` resource.
2. Point it to the same DB used by this project (`DATABASE_URL`).
3. Test connection.

## 2) Create queries

Copy SQL from:

- `/Users/rickyferrer/Working Files/Codex Project/sql/retool_queries.sql`

Create four queries:

1. `portfolio_summary`
2. `campaign_table`
3. `campaign_trend`
4. `risk_alert_feed`

## 3) Build widgets

1. KPI cards bound to `portfolio_summary`.
2. Main table bound to `campaign_table`.
3. Trend line chart bound to `campaign_trend`.
4. Risk-only table bound to `risk_alert_feed`.
5. Add an `advertiserFilter` select input and `campaignIdInput` text/select input.

## 4) Trigger updates

Use a scheduler to run ingestion daily before business hours.

Example cron (Mon-Fri 7:00 AM CT):

```bash
0 7 * * 1-5 cd "/Users/rickyferrer/Working Files/Codex Project" && /usr/bin/python3 src/main.py --csv "/absolute/path/to/latest_export.csv" >> /tmp/dmag_pacing.log 2>&1
```

## 5) Important data requirement

For real pacing alerts, the CSV must include delivered impressions (for example a column like `Delivered` or `Delivered Impressions`) from GAM export data.  
If delivered impressions are missing, the script defaults delivery to `0`, which will overstate risk.
