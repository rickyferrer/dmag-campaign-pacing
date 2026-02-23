import os
from datetime import datetime
from html import escape

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="GAM Pacing Report", page_icon=":bar_chart:", layout="wide")

RISK_COLORS = {
    "high": "#d4183d",
    "medium": "#f59e0b",
    "on_track": "#22c55e",
}

PACE_COLORS = {
    "At Risk": "#ef4444",
    "Behind": "#f97316",
    "Slightly Behind": "#f59e0b",
    "On Track": "#22c55e",
    "Ahead": "#3b82f6",
    "Completed": "#94a3b8",
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f8f9fb; }
        .block-container { padding-top: 4.5rem; max-width: 1440px; }
        @media (max-width: 768px) {
          .block-container { padding-top: 5.25rem; }
        }
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
        .table-shell {
          background: white; border-radius: 12px; border: 1px solid rgba(0,0,0,.08); overflow: hidden; margin-top: 10px;
        }
        .tbl-head, .tbl-row {
          display: grid;
          grid-template-columns: 170px 2.3fr 130px 130px 220px 95px 95px 170px 110px;
          column-gap: 0px;
          align-items: center;
        }
        .tbl-head {
          background: #f8fafc;
          border-bottom: 1px solid #e5e7eb;
          font-weight: 700;
          color: #111827;
        }
        .tbl-cell {
          padding: 12px;
          border-bottom: 1px solid #f1f5f9;
          color: #111827;
        }
        .tbl-head .tbl-cell {
          border-bottom: none;
          padding-top: 14px;
          padding-bottom: 14px;
        }
        .subtext { color:#6b7280; font-size:12px; margin-top: 2px; }
        .chip {
          display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid transparent;
        }
        .chip-risk { color:#b91c1c; background:#fef2f2; border-color:#fecaca; }
        .chip-behind { color:#c2410c; background:#fff7ed; border-color:#fed7aa; }
        .chip-slight { color:#b45309; background:#fffbeb; border-color:#fde68a; }
        .chip-ontrack { color:#166534; background:#f0fdf4; border-color:#bbf7d0; }
        .chip-ahead { color:#1d4ed8; background:#eff6ff; border-color:#bfdbfe; }
        .chip-complete { color:#475569; background:#f8fafc; border-color:#e2e8f0; }
        .pace-wrap { position: relative; width: 180px; height: 12px; background: #e5e7eb; border-radius: 999px; }
        .pace-fill { position: absolute; left:0; top:0; height: 12px; border-radius: 999px; }
        .pace-marker { position: absolute; top: -2px; width: 2px; height: 16px; background: #475569; border-radius: 2px; opacity: .8; }
        .num { font-weight: 600; }
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
        expected_to_date,
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


def pace_state(row: pd.Series) -> str:
    today = pd.Timestamp.now().date()
    end_date = row["end_date"].date() if pd.notna(row["end_date"]) else None
    if end_date and end_date < today:
        return "Completed"
    p = float(row["pacing_pct"]) if pd.notna(row["pacing_pct"]) else 1.0
    if p < 0.50:
        return "At Risk"
    if p < 0.75:
        return "Behind"
    if p < 0.90:
        return "Slightly Behind"
    if p <= 1.15:
        return "On Track"
    return "Ahead"


def chip_class(state: str) -> str:
    return {
        "At Risk": "chip-risk",
        "Behind": "chip-behind",
        "Slightly Behind": "chip-slight",
        "On Track": "chip-ontrack",
        "Ahead": "chip-ahead",
        "Completed": "chip-complete",
    }.get(state, "chip-complete")


def fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return f"{int(n):,}"


def render_custom_table(view: pd.DataFrame) -> None:
    if view.empty:
        st.info("No campaigns match current filters.")
        return

    rows = []
    today = pd.Timestamp.now().date()
    for _, r in view.iterrows():
        state = pace_state(r)
        bar_color = PACE_COLORS[state]
        delivery_pct = max(min(float(r["delivery_pct_of_goal"]), 100.0), 0.0)
        expected_pct = max(min((float(r["expected_to_date"]) / max(float(r["goal_impressions"]), 1)) * 100.0, 100.0), 0.0)
        end_date = r["end_date"].date() if pd.notna(r["end_date"]) else None
        start_date = r["start_date"].date() if pd.notna(r["start_date"]) else None
        if end_date and end_date >= today:
            days_left = (end_date - today).days
            end_label = f"{days_left}d left"
        else:
            end_label = "Ended"
        flight = f"{start_date.strftime('%b %-d') if start_date else '-'} – {end_date.strftime('%b %-d') if end_date else '-'}"

        rows.append(
            (
                f'<div class="tbl-row">'
                f'<div class="tbl-cell"><span class="chip {chip_class(state)}">{escape(state)}</span></div>'
                f'<div class="tbl-cell"><div>{escape(str(r["campaign_name"]))}</div><div class="subtext">{escape(str(r["campaign_id"]))}</div></div>'
                f'<div class="tbl-cell num">{fmt_num(float(r["goal_impressions"]))}</div>'
                f'<div class="tbl-cell num">{fmt_num(float(r["delivered_impressions"]))}</div>'
                f'<div class="tbl-cell"><div class="pace-wrap"><div class="pace-fill" style="width:{delivery_pct:.1f}%; background:{bar_color};"></div><div class="pace-marker" style="left:{expected_pct:.1f}%;"></div></div></div>'
                f'<div class="tbl-cell">{delivery_pct:.1f}%</div>'
                f'<div class="tbl-cell">{expected_pct:.1f}%</div>'
                f'<div class="tbl-cell">{escape(flight)}</div>'
                f'<div class="tbl-cell">{escape(end_label)}</div>'
                f'</div>'
            )
        )

    html = (
        '<div class="table-shell">'
        '<div class="tbl-head">'
        '<div class="tbl-cell">Status</div>'
        '<div class="tbl-cell">Order / Line Item</div>'
        '<div class="tbl-cell">Goal</div>'
        '<div class="tbl-cell">Delivered</div>'
        '<div class="tbl-cell">Pacing</div>'
        '<div class="tbl-cell">Actual %</div>'
        '<div class="tbl-cell">Expected</div>'
        '<div class="tbl-cell">Flight</div>'
        '<div class="tbl-cell">End Date</div>'
        "</div>"
        + "".join(rows)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


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
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

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

    at_risk = int((df["risk_level"] == "high").sum())
    st.markdown(
        f"""<div class="banner">⚠️ {at_risk} campaigns are currently at high risk.</div>""",
        unsafe_allow_html=True,
    )

    total_goal = int(df["goal_impressions"].sum())
    total_delivered = int(df["delivered_impressions"].sum())
    high_count = int((df["risk_level"] == "high").sum())
    med_count = int((df["risk_level"] == "medium").sum())
    ontrack_count = int((df["risk_level"] == "on_track").sum())

    k1, k2, k3, k4 = st.columns(4)
    cards = [
        ("High Risk", f"{high_count}", RISK_COLORS["high"]),
        ("Medium Risk", f"{med_count}", RISK_COLORS["medium"]),
        ("On Track", f"{ontrack_count}", RISK_COLORS["on_track"]),
        ("Delivered / Goal", f"{(total_delivered / max(total_goal, 1)):.1%}", "#030213"),
    ]
    for col, (label, value, color) in zip((k1, k2, k3, k4), cards):
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
    risk_filter = right.multiselect("Risk", ["high", "medium", "on_track"], default=["high", "medium"])
    advertisers = sorted([a for a in df["advertiser"].dropna().unique() if a])
    adv_filter = right2.multiselect("Advertiser", advertisers)
    hide_ended = st.checkbox("Hide ended campaigns", value=True)

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
    if hide_ended:
        today = pd.Timestamp.now().normalize()
        view = view[(view["end_date"].isna()) | (view["end_date"] >= today)]

    # Default: campaigns ending soonest first. Allow user override.
    s1, s2 = st.columns([1.6, 1.0])
    sort_column_label = s1.selectbox(
        "Sort by",
        [
            "End Date (Soonest First)",
            "Campaign Name",
            "Status",
            "Goal",
            "Delivered",
            "Actual %",
            "Expected %",
        ],
        index=0,
    )
    sort_direction = s2.selectbox("Direction", ["Ascending", "Descending"], index=0)

    sort_column_map = {
        "End Date (Soonest First)": "end_date",
        "Campaign Name": "campaign_name",
        "Status": "status",
        "Goal": "goal_impressions",
        "Delivered": "delivered_impressions",
        "Actual %": "delivery_pct_of_goal",
        "Expected %": "expected_to_date",
    }
    sort_col = sort_column_map[sort_column_label]
    asc = sort_direction == "Ascending"
    view = view.sort_values(sort_col, ascending=asc, na_position="last")

    st.markdown(
        f"##### Line Item Details  \n<span class='subtle'>{len(view)} of {len(df)} campaigns</span>",
        unsafe_allow_html=True,
    )
    render_custom_table(view.head(200))

    st.markdown(
        """
        <div class="legend">
          <span><span class="dot" style="background:#d4183d"></span>High</span>
          <span><span class="dot" style="background:#f59e0b"></span>Medium</span>
          <span><span class="dot" style="background:#22c55e"></span>On Track</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
