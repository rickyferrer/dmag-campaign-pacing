import os
from datetime import datetime
from html import escape

import pandas as pd
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
          background: transparent; border: none; border-radius: 0;
          padding: 0; margin: 0;
          box-shadow: none;
        }
        .title-wrap { display:flex; align-items:center; gap:12px; }
        .title-icon {
          width: 44px; height: 44px; border-radius: 14px; background: #030213;
          color: white; display:flex; align-items:center; justify-content:center;
          font-size: 18px;
        }
        .main-title { font-size: 44px; font-weight: 700; color:#111827; line-height: 1.05; }
        .header-subtle { color:#6b7280; font-size:14px; display:flex; align-items:center; gap:8px; margin-top:4px; }
        .help-dot {
          width: 28px; height: 28px; border: 1px solid #d1d5db; border-radius: 999px;
          color:#6b7280; display:flex; align-items:center; justify-content:center; font-weight:700; margin-top: 10px;
        }
        .top-divider { border-bottom:1px solid #e5e7eb; margin:12px -1rem 14px -1rem; }
        .header-gap { height: 18px; }
        .cards-gap { height: 22px; }
        .subtle { color:#717182; font-size:12px; }
        .details-head {
          display:flex; justify-content:space-between; align-items:center; margin-top:16px; margin-bottom:8px;
        }
        .details-left { display:flex; align-items:center; gap:10px; }
        .details-title { font-size:36px; font-weight:700; color:#27272a; line-height:1; }
        .details-badge {
          background:#e5e7eb; color:#6b7280; border-radius:999px; padding:4px 10px; font-size:12px; font-weight:600;
        }
        .details-total { color:#6b7280; font-size:18px; font-weight:500; }
        .details-total b { color:#27272a; }
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
        .th-link {
          color:#111827; text-decoration:none; font-weight:700;
        }
        .th-link:hover { text-decoration:underline; }
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

        /* Filter row styling to match design */
        div[data-testid="stTextInput"] input {
          background:#f3f4f6 !important;
          border:1px solid #e5e7eb !important;
          border-radius:12px !important;
          min-height:46px !important;
          font-size:16px !important;
        }
        div[data-testid="stSelectbox"] > div > div {
          background:#f3f4f6 !important;
          border:1px solid #e5e7eb !important;
          border-radius:12px !important;
          min-height:46px !important;
        }
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
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'campaign_pacing_snapshot';
                """
            )
            existing_cols = {row[0] for row in cur.fetchall()}

    revenue_col = "revenue" if "revenue" in existing_cols else "NULL::double precision AS revenue"
    ecpm_col = "ecpm" if "ecpm" in existing_cols else "NULL::double precision AS ecpm"
    viewability_col = "viewability" if "viewability" in existing_cols else "NULL::double precision AS viewability"

    query = f"""
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
        risk_reason,
        {revenue_col},
        {ecpm_col},
        {viewability_col}
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
def load_latest_overview(database_url: str) -> pd.DataFrame:
    query = """
    SELECT
      snapshot_ts,
      source_report_id,
      impressions_30d,
      viewability_30d,
      impressions_prev_30d,
      viewability_prev_30d
    FROM campaign_overview_snapshot
    ORDER BY snapshot_ts DESC
    LIMIT 2;
    """
    with psycopg.connect(database_url) as conn:
        return pd.read_sql(query, conn)


def format_delta(curr: float, prev: float, suffix: str = "") -> str:
    if prev == 0:
        return "n/a vs previous period"
    abs_delta = curr - prev
    pct_delta = (abs_delta / prev) * 100.0
    sign = "+" if abs_delta >= 0 else "-"
    if suffix:
        abs_text = f"{abs(abs_delta):.1f}"
    else:
        abs_text = fmt_compact_number(abs(abs_delta))
    return f"{sign}{abs_text}{suffix} ({sign}{abs(pct_delta):.1f}%) vs previous period"


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


def fmt_compact_number(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def header_link(label: str, key: str, current_key: str, current_dir: str) -> str:
    is_active = current_key == key
    arrow = ""
    if is_active:
        arrow = " ↑" if current_dir == "asc" else " ↓"
    next_dir = "desc" if (is_active and current_dir == "asc") else "asc"
    return f'<a class="th-link" href="?sort={key}&dir={next_dir}">{escape(label)}{arrow}</a>'


def render_custom_table(view: pd.DataFrame, sort_key: str, sort_dir: str) -> None:
    if view.empty:
        st.info("No campaigns match current filters.")
        return

    today = pd.Timestamp.now().date()
    rows = []
    for _, r in view.iterrows():
        state = pace_state(r)
        delivery_pct = max(min(float(r["delivery_pct_of_goal"]), 100.0), 0.0)
        expected_pct = max(min((float(r["expected_to_date"]) / max(float(r["goal_impressions"]), 1)) * 100.0, 100.0), 0.0)
        bar_color = PACE_COLORS.get(state, "#94a3b8")
        end_date = r["end_date"].date() if pd.notna(r["end_date"]) else None
        start_date = r["start_date"].date() if pd.notna(r["start_date"]) else None
        end_class = ""
        if end_date and end_date >= today:
            days_left = (end_date - today).days
            end_label = f"{days_left}d left"
            if days_left <= 5:
                end_class = ' style="color:#dc2626;font-weight:700;"'
        else:
            end_label = "Ended"
            end_class = ' style="color:#6b7280;"'
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
                f'<div class="tbl-cell"{end_class}>{escape(end_label)}</div>'
                f'</div>'
            )
        )

    html = (
        '<div class="table-shell">'
        '<div class="tbl-head">'
        f'<div class="tbl-cell">{header_link("Status", "status", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Order / Line Item", "campaign_name", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Goal", "goal", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Delivered", "delivered", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Pacing", "actual_pct", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Actual %", "actual_pct", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Expected", "expected_pct", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("Flight", "start_date", sort_key, sort_dir)}</div>'
        f'<div class="tbl-cell">{header_link("End Date", "end_date", sort_key, sort_dir)}</div>'
        '</div>'
        + "".join(rows)
        + '</div>'
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

    try:
        overview_df = load_latest_overview(database_url)
    except Exception:
        overview_df = pd.DataFrame()

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
    df["pace_state"] = df.apply(pace_state, axis=1)

    last_updated = df["snapshot_ts"].max()
    last_updated_text = (
        last_updated.strftime("%b %d, %Y at %I:%M %p") if pd.notna(last_updated) else datetime.now().strftime("%b %d, %Y at %I:%M %p")
    )

    top_a, top_b = st.columns([0.73, 0.27])
    with top_a:
        st.markdown(
            f"""
            <div class="top-shell">
              <div class="title-wrap">
                <div class="title-icon">📊</div>
                <div>
                  <div class="main-title">GAM Pacing Report</div>
                  <div class="header-subtle">🕒 Last updated: {last_updated_text}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_b:
        ecol, rcol = st.columns([0.55, 0.45])
        csv = df.to_csv(index=False).encode("utf-8")
        ecol.download_button("Export CSV", csv, file_name=f"pacing-report-{datetime.now().date()}.csv", use_container_width=True)
        if rcol.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown('<div class="top-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="header-gap"></div>', unsafe_allow_html=True)

    at_risk = int((df["risk_level"] == "high").sum())
    st.markdown(
        f"""<div class="banner">⚠️ {at_risk} campaigns are currently at high risk.</div>""",
        unsafe_allow_html=True,
    )

    total_goal = int(df["goal_impressions"].sum())
    total_delivered = int(df["delivered_impressions"].sum())
    impressions_30d = total_delivered
    viewability_30d = None
    previous_impressions_30d = None
    previous_viewability_30d = None
    if not overview_df.empty:
        latest_row = overview_df.iloc[0]
        impressions_30d = int(latest_row["impressions_30d"]) if pd.notna(latest_row["impressions_30d"]) else impressions_30d
        viewability_30d = float(latest_row["viewability_30d"]) if pd.notna(latest_row["viewability_30d"]) else None
        if "impressions_prev_30d" in latest_row and pd.notna(latest_row["impressions_prev_30d"]):
            previous_impressions_30d = int(latest_row["impressions_prev_30d"])
        if "viewability_prev_30d" in latest_row and pd.notna(latest_row["viewability_prev_30d"]):
            previous_viewability_30d = float(latest_row["viewability_prev_30d"])
        if len(overview_df) > 1:
            prev_row = overview_df.iloc[1]
            if previous_impressions_30d is None and pd.notna(prev_row["impressions_30d"]):
                previous_impressions_30d = int(prev_row["impressions_30d"])
            if previous_viewability_30d is None and pd.notna(prev_row["viewability_30d"]):
                previous_viewability_30d = float(prev_row["viewability_30d"])
    high_count = int((df["risk_level"] == "high").sum())
    ontrack_count = int((df["risk_level"] == "on_track").sum())

    impressions_delta = None
    if previous_impressions_30d is not None:
        impressions_delta = format_delta(float(impressions_30d), float(previous_impressions_30d))

    viewability_delta = None
    if viewability_30d is not None and previous_viewability_30d is not None:
        viewability_delta = format_delta(float(viewability_30d), float(previous_viewability_30d), suffix="pp")

    cards = [
        (
            "Impressions (Last 30 Days)",
            f"{fmt_compact_number(float(impressions_30d))}",
            "#030213",
            impressions_delta or "n/a vs previous period",
        ),
        (
            "Viewability (Last 30 Days)",
            f"{f'{viewability_30d:.1f}%' if viewability_30d is not None else 'N/A'}",
            "#030213",
            viewability_delta or "n/a vs previous period",
        ),
        ("High Risk", f"{high_count}", RISK_COLORS["high"], ""),
        ("On Track", f"{ontrack_count}", RISK_COLORS["on_track"], ""),
        ("Delivered / Goal", f"{(total_delivered / max(total_goal, 1)):.1%}", "#030213", ""),
    ]
    kpi_cols = st.columns(len(cards))
    for col, (label, value, color, delta_text) in zip(kpi_cols, cards):
        with col:
            st.markdown(
                f"""<div class="kpi"><div class="label">{label}</div><div class="value" style="color:{color};">{value}</div><div class="subtle">{delta_text}</div></div>""",
                unsafe_allow_html=True,
            )
    st.markdown('<div class="cards-gap"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="details-head">
          <div class="details-left">
            <div class="details-title">Line Item Details</div>
            <span class="details-badge">{len(df)} of {len(df)}</span>
          </div>
          <div class="details-total">Total Delivered: <b>{fmt_compact_number(float(total_delivered))}</b> / <b>{fmt_compact_number(float(total_goal))}</b> goal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c4, c6 = st.columns([2.8, 1.8, 1.6, 1.0])
    search = c1.text_input(
        "Search orders or line items",
        placeholder="Search orders or line items...",
        label_visibility="collapsed",
    )
    advertisers = sorted([a for a in df["advertiser"].dropna().unique() if a])
    adv_choice = c2.selectbox(
        "All Advertisers",
        ["All Advertisers"] + advertisers,
        index=0,
        label_visibility="collapsed",
    )
    status_choice = c4.selectbox(
        "All Statuses",
        ["All Statuses", "At Risk", "Behind", "Slightly Behind", "On Track", "Ahead", "Completed"],
        index=0,
        label_visibility="collapsed",
    )
    hide_completed = c6.checkbox("Hide completed", value=True)

    view = df.copy()
    if search:
        s = search.lower()
        view = view[
            view["campaign_name"].str.lower().str.contains(s, na=False)
            | view["campaign_id"].astype(str).str.contains(s, na=False)
            | view["advertiser"].astype(str).str.lower().str.contains(s, na=False)
        ]
    if adv_choice != "All Advertisers":
        view = view[view["advertiser"] == adv_choice]
    if status_choice != "All Statuses":
        view = view[view["pace_state"] == status_choice]
    if hide_completed:
        view = view[view["pace_state"] != "Completed"]

    qp = st.query_params
    requested_sort = str(qp.get("sort", "end_date"))
    requested_dir = str(qp.get("dir", "asc")).lower()
    if requested_dir not in {"asc", "desc"}:
        requested_dir = "asc"

    sort_map = {
        "status": "pace_state",
        "campaign_name": "campaign_name",
        "goal": "goal_impressions",
        "delivered": "delivered_impressions",
        "actual_pct": "delivery_pct_of_goal",
        "expected_pct": "expected_to_date",
        "start_date": "start_date",
        "end_date": "end_date",
    }
    if requested_sort not in sort_map:
        requested_sort = "end_date"
    sort_col = sort_map[requested_sort]
    ascending = requested_dir == "asc"
    view = view.sort_values(sort_col, ascending=ascending, na_position="last")

    render_custom_table(view.head(200), requested_sort, requested_dir)

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
