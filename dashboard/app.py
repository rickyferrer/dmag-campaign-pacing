import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="GAM Pacing Report", page_icon=":bar_chart:", layout="wide")

RISK_COLORS = {
    "high": "#d4183d",
    "medium": "#f59e0b",
    "watch": "#eab308",
    "on_track": "#22c55e",
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f8f9fb; }
        .block-container { padding-top: 1.25rem; max-width: 1440px; }
        .top-shell {
          background: white; border: 1px solid rgba(0,0,0,.08); border-radius: 12px;
          padding: 14px 18px; margin-bottom: 12px;
          box-shadow: 0 1px 2px rgba(15,23,42,.05);
        }
        .title-wrap { display:flex; align-items:center; gap:12px; }
        .title-icon {
          width: 32px; height: 32px; border-radius: 10px; background: #030213;
          color: white; display:flex; align-items:center; justify-content:center;
          font-size: 16px;
        }
        .subtle { color:#717182; font-size:12px; }
        .banner {
          border-radius: 10px; padding: 10px 14px; margin: 8px 0 14px 0;
          border: 1px solid #fee2e2; background: #fff5f7; color: #9f1239;
          font-size: 13px; font-weight: 500;
        }
        .kpi {
          background: white; border-radius: 12px; border: 1px solid rgba(0,0,0,.08);
          padding: 14px 16px; min-height: 94px;
        }
        .kpi .label { color: #717182; font-size: 12px; }
        .kpi .value { color: #030213; font-size: 26px; font-weight: 700; line-height: 1.1; margin-top: 4px; }
        .panel {
          background: white; border-radius: 12px; border: 1px solid rgba(0,0,0,.08);
          padding: 8px 12px 6px 12px; margin-top: 10px;
        }
        .legend {
          margin-top: 8px; color: #717182; font-size: 12px; display: flex; gap: 14px; flex-wrap: wrap;
        }
        .dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; margin-right: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_database_url() -> str:
    try:
        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"]).strip()
    except Exception:
        pass
    load_dotenv(override=False)
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


def main() -> None:
    apply_theme()
    database_url = get_database_url()
    if not database_url:
        st.error("DATABASE_URL is missing. Set it in Streamlit Secrets or .env.")
        st.stop()

    try:
        df = load_latest_campaigns(database_url)
    except Exception as exc:
        st.error(f"Failed to load campaign data: {exc}")
        st.stop()

    if df.empty:
        st.warning("No campaign snapshots found yet. Run the ingestion script first.")
        st.stop()

    df["risk_level"] = df["risk_level"].fillna("on_track")
    df["status"] = df["status"].fillna("")
    df["projected_pct_of_goal"] = (df["projected_final"] / df["goal_impressions"]) * 100
    df["delivery_pct_of_goal"] = (df["delivered_impressions"] / df["goal_impressions"]) * 100
    df["gap_to_goal"] = (df["goal_impressions"] - df["projected_final"]).clip(lower=0)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")

    last_updated = df["snapshot_ts"].max()
    last_updated_text = (
        last_updated.strftime("%b %d, %Y at %I:%M %p") if pd.notna(last_updated) else datetime.now().strftime("%b %d, %Y at %I:%M %p")
    )

    top_a, top_b = st.columns([0.7, 0.3])
    with top_a:
        st.markdown(
            f"""
            <div class="top-shell">
              <div class="title-wrap">
                <div class="title-icon">📊</div>
                <div>
                  <div style="font-size:22px; font-weight:700; color:#030213;">GAM Pacing Report</div>
                  <div class="subtle">Last updated: {last_updated_text}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_b:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv, file_name=f"pacing-report-{datetime.now().date()}.csv", use_container_width=True)
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    at_risk = int((df["risk_level"].isin(["high", "medium"])).sum())
    st.markdown(
        f"""<div class="banner">⚠️ {at_risk} campaigns are currently at risk (high/medium).</div>""",
        unsafe_allow_html=True,
    )

    total_goal = int(df["goal_impressions"].sum())
    total_delivered = int(df["delivered_impressions"].sum())
    high_count = int((df["risk_level"] == "high").sum())
    med_count = int((df["risk_level"] == "medium").sum())
    watch_count = int((df["risk_level"] == "watch").sum())
    ontrack_count = int((df["risk_level"] == "on_track").sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        ("High Risk", f"{high_count}", RISK_COLORS["high"]),
        ("Medium Risk", f"{med_count}", RISK_COLORS["medium"]),
        ("Watch", f"{watch_count}", RISK_COLORS["watch"]),
        ("On Track", f"{ontrack_count}", RISK_COLORS["on_track"]),
        ("Delivered / Goal", f"{(total_delivered / max(total_goal, 1)):.1%}", "#030213"),
    ]
    for col, (label, value, color) in zip((k1, k2, k3, k4, k5), cards):
        with col:
            st.markdown(
                f"""<div class="kpi"><div class="label">{label}</div><div class="value" style="color:{color};">{value}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("##### Average Delivery by Advertiser")
    adv = (
        df.groupby("advertiser", dropna=False, as_index=False)
        .agg(delivered=("delivered_impressions", "sum"), goal=("goal_impressions", "sum"))
        .fillna({"advertiser": "Unknown"})
    )
    adv["delivery_pct"] = (adv["delivered"] / adv["goal"].replace(0, pd.NA) * 100).fillna(0)
    adv = adv.sort_values("delivery_pct", ascending=False).head(15)
    fig_adv = px.bar(
        adv,
        x="advertiser",
        y="delivery_pct",
        color="delivery_pct",
        color_continuous_scale=["#e2e8f0", "#64748b", "#030213"],
        labels={"advertiser": "", "delivery_pct": "% Delivered vs Goal"},
    )
    fig_adv.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=10, b=20),
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_adv, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right, right2 = st.columns([2.0, 1.3, 1.3])
    search = left.text_input("Search line item/order", placeholder="Search by campaign name or ID")
    risk_filter = right.multiselect("Risk", ["high", "medium", "watch", "on_track"], default=["high", "medium", "watch"])
    advertisers = sorted([a for a in df["advertiser"].dropna().unique() if a])
    adv_filter = right2.multiselect("Advertiser", advertisers)

    view = df.copy()
    if search:
        s = search.lower()
        view = view[
            view["campaign_name"].str.lower().str.contains(s, na=False)
            | view["campaign_id"].astype(str).str.contains(s, na=False)
            | view["advertiser"].astype(str).str.lower().str.contains(s, na=False)
        ]
    if risk_filter:
        view = view[view["risk_level"].isin(risk_filter)]
    if adv_filter:
        view = view[view["advertiser"].isin(adv_filter)]

    st.markdown(f"##### Line Item Details  \n<span class='subtle'>{len(view)} of {len(df)} campaigns</span>", unsafe_allow_html=True)
    display = view[
        [
            "campaign_id",
            "status",
            "advertiser",
            "campaign_name",
            "start_date",
            "end_date",
            "goal_impressions",
            "delivered_impressions",
            "delivery_pct_of_goal",
            "projected_final",
            "projected_pct_of_goal",
            "required_daily_run_rate",
            "risk_level",
            "risk_reason",
        ]
    ].copy()
    display["delivery_pct_of_goal"] = display["delivery_pct_of_goal"].round(1)
    display["projected_pct_of_goal"] = display["projected_pct_of_goal"].round(1)
    display["projected_final"] = display["projected_final"].round(0).astype(int)
    display["required_daily_run_rate"] = display["required_daily_run_rate"].round(0).astype(int)
    st.dataframe(display, use_container_width=True, height=540)

    st.markdown(
        """
        <div class="legend">
          <span><span class="dot" style="background:#d4183d"></span>High</span>
          <span><span class="dot" style="background:#f59e0b"></span>Medium</span>
          <span><span class="dot" style="background:#eab308"></span>Watch</span>
          <span><span class="dot" style="background:#22c55e"></span>On Track</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
