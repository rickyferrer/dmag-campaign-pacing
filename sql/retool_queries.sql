-- 1) portfolio_summary
WITH latest AS (
    SELECT DISTINCT ON (campaign_id)
        snapshot_ts,
        campaign_id,
        campaign_name,
        advertiser,
        status,
        goal_impressions,
        delivered_impressions,
        pacing_pct,
        projected_final,
        risk_level
    FROM campaign_pacing_snapshot
    ORDER BY campaign_id, snapshot_ts DESC
)
SELECT
    COUNT(*) AS total_campaigns,
    COUNT(*) FILTER (WHERE risk_level = 'high') AS high_risk_campaigns,
    COUNT(*) FILTER (WHERE risk_level = 'medium') AS medium_risk_campaigns,
    COUNT(*) FILTER (WHERE risk_level = 'watch') AS watch_campaigns,
    COUNT(*) FILTER (WHERE risk_level = 'on_track') AS on_track_campaigns,
    SUM(goal_impressions) AS total_goal_impressions,
    SUM(delivered_impressions) AS total_delivered_impressions
FROM latest;

-- 2) campaign_table
WITH latest AS (
    SELECT DISTINCT ON (campaign_id)
        snapshot_ts,
        campaign_id,
        campaign_name,
        advertiser,
        status,
        start_date,
        end_date,
        goal_impressions,
        delivered_impressions,
        pacing_pct,
        projected_final,
        required_daily_run_rate,
        risk_level,
        risk_reason
    FROM campaign_pacing_snapshot
    ORDER BY campaign_id, snapshot_ts DESC
)
SELECT
    snapshot_ts,
    campaign_id,
    campaign_name,
    advertiser,
    status,
    start_date,
    end_date,
    goal_impressions,
    delivered_impressions,
    ROUND(pacing_pct * 100.0, 2) AS pacing_pct,
    projected_final::BIGINT AS projected_final,
    required_daily_run_rate::BIGINT AS required_daily_run_rate,
    risk_level,
    risk_reason
FROM latest
WHERE
    (
        {{ !advertiserFilter.value || advertiser = advertiserFilter.value }}
    )
ORDER BY
    CASE risk_level
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'watch' THEN 3
        ELSE 4
    END,
    pacing_pct ASC;

-- 3) campaign_trend
SELECT
    DATE(snapshot_ts) AS snapshot_date,
    campaign_id,
    campaign_name,
    advertiser,
    status,
    delivered_impressions,
    expected_to_date::BIGINT AS expected_to_date,
    projected_final::BIGINT AS projected_final,
    risk_level
FROM campaign_pacing_snapshot
WHERE campaign_id = {{ campaignIdInput.value }}
ORDER BY snapshot_date ASC;

-- 4) risk_alert_feed (latest risky only)
WITH latest AS (
    SELECT DISTINCT ON (campaign_id)
        snapshot_ts,
        campaign_id,
        campaign_name,
        advertiser,
        status,
        goal_impressions,
        delivered_impressions,
        projected_final,
        required_daily_run_rate,
        risk_level,
        risk_reason
    FROM campaign_pacing_snapshot
    ORDER BY campaign_id, snapshot_ts DESC
)
SELECT
    snapshot_ts,
    campaign_id,
    campaign_name,
    advertiser,
    status,
    goal_impressions,
    delivered_impressions,
    projected_final::BIGINT AS projected_final,
    required_daily_run_rate::BIGINT AS required_daily_run_rate,
    risk_level,
    risk_reason
FROM latest
WHERE
    risk_level IN ('high', 'medium', 'watch')
    AND LOWER(COALESCE(status, '')) IN ('live', 'ready', 'no assets', 'paused')
ORDER BY
    CASE risk_level
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        ELSE 3
    END,
    projected_final / NULLIF(goal_impressions, 0) ASC;
