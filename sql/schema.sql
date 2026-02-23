CREATE TABLE IF NOT EXISTS campaign_pacing_snapshot (
    id BIGSERIAL PRIMARY KEY,
    snapshot_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    campaign_id TEXT NOT NULL,
    campaign_name TEXT NOT NULL,
    advertiser TEXT,
    status TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    goal_impressions BIGINT NOT NULL,
    delivered_impressions BIGINT NOT NULL,
    days_elapsed INTEGER NOT NULL,
    total_days INTEGER NOT NULL,
    days_remaining INTEGER NOT NULL,
    expected_to_date DOUBLE PRECISION NOT NULL,
    pacing_pct DOUBLE PRECISION NOT NULL,
    projected_final DOUBLE PRECISION NOT NULL,
    required_daily_run_rate DOUBLE PRECISION NOT NULL,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('on_track', 'watch', 'medium', 'high')),
    risk_reason TEXT NOT NULL,
    revenue DOUBLE PRECISION,
    ecpm DOUBLE PRECISION,
    viewability DOUBLE PRECISION
);

ALTER TABLE campaign_pacing_snapshot
ADD COLUMN IF NOT EXISTS revenue DOUBLE PRECISION;

ALTER TABLE campaign_pacing_snapshot
ADD COLUMN IF NOT EXISTS ecpm DOUBLE PRECISION;

ALTER TABLE campaign_pacing_snapshot
ADD COLUMN IF NOT EXISTS viewability DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_campaign_pacing_snapshot_ts
ON campaign_pacing_snapshot (snapshot_ts DESC);

CREATE INDEX IF NOT EXISTS idx_campaign_pacing_snapshot_campaign_ts
ON campaign_pacing_snapshot (campaign_id, snapshot_ts DESC);

CREATE INDEX IF NOT EXISTS idx_campaign_pacing_snapshot_risk
ON campaign_pacing_snapshot (risk_level, snapshot_ts DESC);

CREATE TABLE IF NOT EXISTS campaign_overview_snapshot (
    id BIGSERIAL PRIMARY KEY,
    snapshot_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_report_id TEXT NOT NULL,
    impressions_30d BIGINT NOT NULL,
    viewability_30d DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_campaign_overview_snapshot_ts
ON campaign_overview_snapshot (snapshot_ts DESC);
