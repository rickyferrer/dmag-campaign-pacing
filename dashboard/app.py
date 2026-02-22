import os

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st
from dotenv import load_dotenv


st.set_page_config(page_title="D Magazine Campaign Pacing", layout="wide")


def get_database_url() -> str:
    # Streamlit Cloud secrets first; local .env fallback.
    try:
        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"]).strip()
    except Exception:
        pass

    load_dotenv(override=True)
    return os.getenv("DATABASE_URL", "").strip()


@st.cache_data(ttl=300)
def load_latest_campaigns(database_url: str) -> pd.DataFrame:
    query = """
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
        projected_final,
        required_daily_run_rate,
        pacing_pct,
        risk_level,
        risk_reason
      FROM campaign_pacing_snapshot
      ORDER BY campaign_id, snapshot_ts DESC
    )
    SELECT *
    FROM latest
    ORDER BY
      CASE risk_level
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'watch' THEN 3
        ELSE 4
      END,
      projected_final / NULLIF(goal_impressions, 0) ASC;
    """
    with psycopg.connect(database_url) as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=300)
def load_history(database_url: str) -> pd.DataFrame:
    query = """
    SELECT
      DATE(snapshot_ts) AS snapshot_date,
      risk_level,
      COUNT(*)::INT AS campaign_count
    FROM campaign_pacing_snapshot
    GROUP BY DATE(snapshot_ts), risk_level
    ORDER BY snapshot_date ASC;
    """
    with psycopg.connect(database_url) as conn:
        return pd.read_sql(query, conn)


def main() -> None:
    database_url = get_database_url()
    if not database_url:
        st.error("DATABASE_URL is missing. Set it in Streamlit Secrets or .env.")
        st.stop()

    st.title("D Magazine Ad Campaign Pacing")
    st.caption("Latest snapshot by campaign from GAM pacing pipeline")

    try:
        df = load_latest_campaigns(database_url)
    except Exception as exc:
        st.error(f"Failed to load campaign data: {exc}")
        st.stop()

    if df.empty:
        st.warning("No campaign snapshots found yet. Run the ingestion script first.")
        st.stop()

    df["projected_pct_of_goal"] = (df["projected_final"] / df["goal_impressions"]) * 100
    df["gap_to_goal"] = (df["goal_impressions"] - df["projected_final"]).clip(lower=0)
    df["risk_level"] = df["risk_level"].fillna("on_track")
    df["status"] = df["status"].fillna("")

    risk_options = ["high", "medium", "watch", "on_track"]
    selected_risk = st.multiselect("Risk filter", risk_options, default=["high", "medium", "watch"])
    advertiser_options = sorted([x for x in df["advertiser"].dropna().unique() if x])
    selected_advertisers = st.multiselect("Advertiser filter", advertiser_options)

    view = df.copy()
    if selected_risk:
        view = view[view["risk_level"].isin(selected_risk)]
    if selected_advertisers:
        view = view[view["advertiser"].isin(selected_advertisers)]

    high_count = int((df["risk_level"] == "high").sum())
    med_count = int((df["risk_level"] == "medium").sum())
    watch_count = int((df["risk_level"] == "watch").sum())
    total_goal = int(df["goal_impressions"].sum())
    total_delivered = int(df["delivered_impressions"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("High Risk", f"{high_count}")
    c2.metric("Medium Risk", f"{med_count}")
    c3.metric("Watch", f"{watch_count}")
    c4.metric("Goal Impressions", f"{total_goal:,}")
    c5.metric("Delivered", f"{total_delivered:,}")

    left, right = st.columns([1, 1])
    with left:
        risk_counts = (
            df.groupby("risk_level", as_index=False)["campaign_id"]
            .count()
            .rename(columns={"campaign_id": "campaigns"})
        )
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="campaigns",
            title="Campaigns by Risk Level",
            color="risk_level",
            color_discrete_map={
                "high": "#d62728",
                "medium": "#ff7f0e",
                "watch": "#bcbd22",
                "on_track": "#2ca02c",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_gap = (
            df[df["risk_level"].isin(["high", "medium"])]
            .sort_values("gap_to_goal", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_gap,
            x="gap_to_goal",
            y="campaign_name",
            orientation="h",
            title="Top 10 Gap to Goal (Projected)",
            color="risk_level",
            color_discrete_map={"high": "#d62728", "medium": "#ff7f0e"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    try:
        hist = load_history(database_url)
        if not hist.empty:
            fig = px.area(
                hist,
                x="snapshot_date",
                y="campaign_count",
                color="risk_level",
                title="Risk Trend Over Time",
                color_discrete_map={
                    "high": "#d62728",
                    "medium": "#ff7f0e",
                    "watch": "#bcbd22",
                    "on_track": "#2ca02c",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.subheader("Campaign Detail")
    detail_cols = [
        "campaign_name",
        "campaign_id",
        "advertiser",
        "status",
        "start_date",
        "end_date",
        "risk_level",
        "goal_impressions",
        "delivered_impressions",
        "projected_final",
        "projected_pct_of_goal",
        "required_daily_run_rate",
        "risk_reason",
    ]
    display = view[detail_cols].copy()
    display["projected_final"] = display["projected_final"].round(0).astype(int)
    display["projected_pct_of_goal"] = display["projected_pct_of_goal"].round(1)
    display["required_daily_run_rate"] = display["required_daily_run_rate"].round(0).astype(int)
    st.dataframe(display, use_container_width=True, height=520)


if __name__ == "__main__":
    main()
