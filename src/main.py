import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from email.mime.text import MIMEText
import smtplib
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(override=True) -> None:
        return None


DEFAULT_COLUMNS = {
    "campaign_id": "campaign_id",
    "campaign_name": "campaign_name",
    "advertiser": "advertiser",
    "start_date": "start_date",
    "end_date": "end_date",
    "goal_impressions": "goal_impressions",
    "delivered_impressions": "delivered_impressions",
}


@dataclass
class CampaignMetrics:
    campaign_id: str
    campaign_name: str
    advertiser: str
    status: str
    start_date: date
    end_date: date
    goal_impressions: int
    delivered_impressions: int
    days_elapsed: int
    total_days: int
    days_remaining: int
    expected_to_date: float
    pacing_pct: float
    projected_final: float
    required_daily_run_rate: float
    risk_level: str
    risk_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campaign pacing ingestion and alerting")
    parser.add_argument("--source", choices=["csv", "gam"], default="csv", help="Input source")
    parser.add_argument("--csv", help="Absolute path to pacing CSV export")
    parser.add_argument("--gam-report-id", help="Ad Manager interactive report ID")
    parser.add_argument("--dry-run", action="store_true", help="Skip DB write and email send")
    parser.add_argument("--skip-email", action="store_true", help="Write snapshot but do not send email")
    args = parser.parse_args()
    if args.source == "csv" and not args.csv:
        parser.error("--csv is required when --source=csv")
    if args.source == "gam" and not args.gam_report_id:
        parser.error("--gam-report-id is required when --source=gam")
    return args


def parse_date(raw: str) -> date:
    raw = raw.strip()
    if raw == "" or raw.lower() in {"unlimited", "n/a", "na", "none"}:
        raise ValueError(f"Unsupported date format: {raw}")
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw}")


def parse_int(raw: str) -> int:
    clean = re.sub(r"[^0-9.-]", "", raw.strip())
    if clean == "":
        return 0
    return int(float(clean))


def load_column_map() -> Dict[str, str]:
    custom = os.getenv("CSV_COLUMN_MAP", "").strip()
    if not custom:
        return DEFAULT_COLUMNS
    mapped = json.loads(custom)
    merged = DEFAULT_COLUMNS.copy()
    merged.update(mapped)
    return merged


def load_gam_field_map() -> Dict[str, int]:
    default_map = {
        # Expected ordering if report columns mirror your sheet layout.
        "campaign_name": 0,  # CLIENT
        "campaign_id": 2,  # IO
        "status": 4,  # STATUS
        "start_date": 5,  # START
        "end_date": 6,  # END
        "delivered_impressions": 8,  # IMPRESSIONS
        "goal_impressions": 9,  # GOAL
    }
    custom = os.getenv("GAM_FIELD_MAP", "").strip()
    if not custom:
        return default_map
    parsed = json.loads(custom)
    merged = default_map.copy()
    merged.update({k: int(v) for k, v in parsed.items()})
    return merged


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().strip())


def candidate_header_matches(row: List[str]) -> bool:
    if not row:
        return False
    normalized = {normalize_column(x) for x in row if x.strip()}
    # Flexible matching for expected pacing fields.
    signals = {
        "advertiser",
        "client",
        "order",
        "campaignid",
        "campaignname",
        "lineitemid",
        "lineitem",
        "lineitemstartdate",
        "lineitemenddate",
        "adunitstatus",
        "lineitemprimarygoalunitsabsolute",
        "adserverimpressions",
        "impressionsgoal",
        "impressions",
        "impgoal",
        "goalimpressions",
        "start",
        "startdate",
        "end",
        "enddate",
    }
    return len(normalized.intersection(signals)) >= 3


def iter_section_rows(csv_path: str):
    current_header: Optional[List[str]] = None
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if candidate_header_matches(row):
                current_header = row
                continue
            if current_header is None:
                continue
            raw = {}
            for i, key in enumerate(current_header):
                if not key.strip():
                    continue
                raw[key] = row[i] if i < len(row) else ""
            if raw:
                yield raw


def best_available_column(row: Dict[str, str], options: List[str]) -> Optional[str]:
    normalized_map = {normalize_column(k): k for k in row.keys()}
    for candidate in options:
        nk = normalize_column(candidate)
        if nk in normalized_map:
            return normalized_map[nk]
    return None


def clamp(n: int, min_v: int, max_v: int) -> int:
    return max(min_v, min(n, max_v))


def compute_metrics(row: Dict[str, str], columns: Dict[str, str], today: date) -> CampaignMetrics:
    advertiser_col = best_available_column(row, [columns["advertiser"], "Advertiser", "Client", "Order"])
    start_col = best_available_column(row, [columns["start_date"], "Start", "Start Date", "Line item start date"])
    end_col = best_available_column(row, [columns["end_date"], "End", "End Date", "Line item end date"])
    goal_col = best_available_column(
        row,
        [columns["goal_impressions"], "Imp. Goal", "Impressions Goal", "Goal", "Line item primary goal units (absolute)"],
    )
    delivered_col = best_available_column(
        row,
        [
            columns["delivered_impressions"],
            "Delivered",
            "Delivered Impressions",
            "Imp. Delivered",
            "Actuals",
            "Impressions",
            "Ad server impressions",
        ],
    )
    status_col = best_available_column(row, ["status", "Status", "Ad unit status", "Line item status"])
    campaign_name_col = best_available_column(
        row,
        [columns["campaign_name"], "Campaign", "Line Item", "Line item", "Advertiser", "Order"],
    )
    campaign_id_col = best_available_column(row, [columns["campaign_id"], "Campaign ID", "Line Item ID", "Line item ID", "IO"])

    if not all([advertiser_col, start_col, end_col, goal_col, delivered_col]):
        raise ValueError("CSV missing one or more required fields (advertiser/start/end/goal/delivered).")

    advertiser = str(row.get(advertiser_col, "")).strip()
    campaign_name = str(row.get(campaign_name_col or advertiser_col, advertiser)).strip() or advertiser
    start_date = parse_date(row[start_col])
    end_date = parse_date(row[end_col])
    goal = parse_int(row[goal_col])
    delivered = parse_int(row[delivered_col])
    status = str(row.get(status_col or "", "")).strip() if status_col else ""
    derived_id = f"{advertiser}|{campaign_name}|{start_date.isoformat()}|{end_date.isoformat()}|{goal}"
    campaign_id = str(row.get(campaign_id_col, "")).strip() if campaign_id_col else derived_id
    if not campaign_id:
        campaign_id = derived_id

    if goal <= 0:
        raise ValueError("Goal is zero or missing; skipping.")

    # Do not flag campaigns as pacing risk before their flight starts.
    if today < start_date:
        total_days = max((end_date - start_date).days + 1, 1)
        days_remaining = total_days
        return CampaignMetrics(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            advertiser=advertiser,
            status=status,
            start_date=start_date,
            end_date=end_date,
            goal_impressions=goal,
            delivered_impressions=delivered,
            days_elapsed=0,
            total_days=total_days,
            days_remaining=days_remaining,
            expected_to_date=0.0,
            pacing_pct=1.0,
            projected_final=float(delivered),
            required_daily_run_rate=goal / max(days_remaining, 1),
            risk_level="on_track",
            risk_reason="Campaign has not started yet.",
        )
    total_days = max((end_date - start_date).days + 1, 1)
    days_elapsed_raw = (today - start_date).days + 1
    days_elapsed = clamp(days_elapsed_raw, 0, total_days)
    days_remaining = max(total_days - days_elapsed, 0)

    expected_to_date = goal * (days_elapsed / total_days) if total_days else float(goal)
    pacing_pct = (delivered / expected_to_date) if expected_to_date > 0 else 1.0

    avg_daily_rate = delivered / max(days_elapsed, 1)
    projected_final = delivered + (avg_daily_rate * days_remaining)
    remaining_goal = max(goal - delivered, 0)
    required_daily_run_rate = remaining_goal / max(days_remaining, 1) if days_remaining > 0 else 0.0

    risk_level = "on_track"
    risk_reason = "Projected delivery is on track."
    if projected_final < goal * 0.90:
        risk_level = "high"
        risk_reason = "Projected final delivery is below 90% of goal."
    elif projected_final < goal * 0.95:
        risk_level = "medium"
        risk_reason = "Projected final delivery is below 95% of goal."
    elif pacing_pct < 0.95:
        risk_level = "watch"
        risk_reason = "Current pacing is below 95% of expected-to-date delivery."

    return CampaignMetrics(
        campaign_id=campaign_id,
        campaign_name=campaign_name,
        advertiser=advertiser,
        status=status,
        start_date=start_date,
        end_date=end_date,
        goal_impressions=goal,
        delivered_impressions=delivered,
        days_elapsed=days_elapsed,
        total_days=total_days,
        days_remaining=days_remaining,
        expected_to_date=expected_to_date,
        pacing_pct=pacing_pct,
        projected_final=projected_final,
        required_daily_run_rate=required_daily_run_rate,
        risk_level=risk_level,
        risk_reason=risk_reason,
    )


def load_csv_metrics(csv_path: str, columns: Dict[str, str]) -> List[CampaignMetrics]:
    today = date.today()
    output: List[CampaignMetrics] = []
    skipped = 0
    for raw in iter_section_rows(csv_path):
        if not any((v or "").strip() for v in raw.values()):
            continue
        try:
            output.append(compute_metrics(raw, columns, today))
        except (ValueError, KeyError):
            skipped += 1
            continue
    deduped: Dict[str, CampaignMetrics] = {}
    status_priority = {"live": 4, "ready": 3, "no assets": 2, "paused": 1}
    for m in output:
        key = f"{m.campaign_id}|{m.start_date.isoformat()}|{m.end_date.isoformat()}|{m.goal_impressions}"
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = m
            continue
        existing_score = (
            status_priority.get(existing.status.strip().lower(), 0),
            existing.delivered_impressions,
        )
        candidate_score = (
            status_priority.get(m.status.strip().lower(), 0),
            m.delivered_impressions,
        )
        if candidate_score >= existing_score:
            deduped[key] = m
    if skipped:
        print(f"Skipped {skipped} rows that did not match pacing schema.")
    return list(deduped.values())


def _pb_value_to_text(value) -> str:
    """Convert protobuf/wrapper values into plain strings."""
    # Simple scalar wrappers often expose ".value".
    if hasattr(value, "value"):
        wrapped = getattr(value, "value")
        if wrapped not in (None, ""):
            return str(wrapped)

    # Date/date-time value objects.
    year = getattr(value, "year", 0)
    month = getattr(value, "month", 0)
    day = getattr(value, "day", 0)
    if year and month and day:
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Generic protobuf message traversal.
    if hasattr(value, "ListFields"):
        fields = value.ListFields()
        if not fields:
            return ""
        # Report values are oneof-like; use first populated field.
        _, nested = fields[0]
        return _pb_value_to_text(nested)

    if value in (None, ""):
        return ""
    return str(value)


def _extract_report_value(cell) -> str:
    # Use protobuf internals directly for the most compatible parsing across
    # generated wrappers.
    pb = getattr(cell, "_pb", cell)
    if not hasattr(pb, "ListFields"):
        return ""
    fields = pb.ListFields()
    if not fields:
        return ""
    _, value = fields[0]
    return _pb_value_to_text(value)


def fetch_gam_report_rows(network_code: str, report_id: str) -> List[List[str]]:
    try:
        from google.ads import admanager_v1
    except ImportError as exc:
        raise RuntimeError(
            "google-ads-admanager is not installed. Run: pip install google-ads-admanager"
        ) from exc

    client = admanager_v1.ReportServiceClient()
    report_name = f"networks/{network_code}/reports/{report_id}"
    run_request = admanager_v1.RunReportRequest(name=report_name)
    operation = client.run_report(request=run_request)
    response = operation.result(timeout=900)
    result_name = response.report_result

    fetch_request = admanager_v1.FetchReportResultRowsRequest(name=result_name)
    page_result = client.fetch_report_result_rows(request=fetch_request)

    rows: List[List[str]] = []
    for row in page_result:
        dimension_values = [_extract_report_value(v) for v in row.dimension_values]
        metric_values = []
        if row.metric_value_groups:
            metric_values = [_extract_report_value(v) for v in row.metric_value_groups[0].primary_values]
        rows.append(dimension_values + metric_values)
    return rows


def load_gam_metrics(report_id: str, columns: Dict[str, str]) -> List[CampaignMetrics]:
    network_code = os.getenv("GAM_NETWORK_CODE", "").strip()
    if not network_code:
        raise RuntimeError("GAM_NETWORK_CODE is required for --source=gam")

    field_map = load_gam_field_map()
    today = date.today()
    skipped = 0
    output: List[CampaignMetrics] = []

    rows = fetch_gam_report_rows(network_code=network_code, report_id=report_id)
    for values in rows:
        try:
            row = {
                "campaign_name": values[field_map["campaign_name"]] if len(values) > field_map["campaign_name"] else "",
                "campaign_id": values[field_map["campaign_id"]] if len(values) > field_map["campaign_id"] else "",
                "advertiser": values[field_map.get("advertiser", field_map["campaign_name"])]
                if len(values) > field_map.get("advertiser", field_map["campaign_name"])
                else "",
                "status": values[field_map["status"]] if len(values) > field_map["status"] else "",
                "start_date": values[field_map["start_date"]] if len(values) > field_map["start_date"] else "",
                "end_date": values[field_map["end_date"]] if len(values) > field_map["end_date"] else "",
                "delivered_impressions": values[field_map["delivered_impressions"]]
                if len(values) > field_map["delivered_impressions"]
                else "",
                "goal_impressions": values[field_map["goal_impressions"]] if len(values) > field_map["goal_impressions"] else "",
            }
            output.append(compute_metrics(row, columns, today))
        except (ValueError, KeyError, IndexError):
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} GAM rows that did not match field map.")
    return output


def write_snapshot_to_db(metrics: List[CampaignMetrics]) -> None:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for DB writes.")

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is not installed. Run: pip install -r requirements.txt") from exc

    rows = [
        (
            m.campaign_id,
            m.campaign_name,
            m.advertiser,
            m.status,
            m.start_date,
            m.end_date,
            m.goal_impressions,
            m.delivered_impressions,
            m.days_elapsed,
            m.total_days,
            m.days_remaining,
            m.expected_to_date,
            m.pacing_pct,
            m.projected_final,
            m.required_daily_run_rate,
            m.risk_level,
            m.risk_reason,
        )
        for m in metrics
    ]

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO campaign_pacing_snapshot (
                    campaign_id,
                    campaign_name,
                    advertiser,
                    status,
                    start_date,
                    end_date,
                    goal_impressions,
                    delivered_impressions,
                    days_elapsed,
                    total_days,
                    days_remaining,
                    expected_to_date,
                    pacing_pct,
                    projected_final,
                    required_daily_run_rate,
                    risk_level,
                    risk_reason
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                );
                """,
                rows,
            )
        conn.commit()


def format_alert_body(metrics: List[CampaignMetrics]) -> str:
    active_statuses = {"live", "ready", "no assets", "paused", "active"}
    today = date.today()
    risky = [
        m
        for m in metrics
        if m.risk_level in {"high", "medium", "watch"}
        and m.status.strip().lower() in active_statuses
        and m.end_date >= today
    ]
    risky.sort(key=lambda x: (x.risk_level, x.projected_final / max(x.goal_impressions, 1)))
    max_rows = int(os.getenv("ALERT_MAX_ROWS", "50"))

    lines = []
    lines.append(f"Campaign pacing alert - {date.today().isoformat()}")
    lines.append("")
    if not risky:
        lines.append("No at-risk campaigns detected.")
    else:
        lines.append(f"At-risk campaigns: {len(risky)}")
        lines.append("")
        for m in risky[:max_rows]:
            ratio = m.projected_final / max(m.goal_impressions, 1)
            lines.append(
                (
                    f"[{m.risk_level.upper()}] {m.campaign_name} ({m.campaign_id}) | "
                    f"Status: {m.status or 'N/A'} | "
                    f"Goal: {m.goal_impressions:,} | Delivered: {m.delivered_impressions:,} | "
                    f"Projected: {int(m.projected_final):,} ({ratio:.1%}) | "
                    f"Required Daily: {int(m.required_daily_run_rate):,} | "
                    f"Reason: {m.risk_reason}"
                )
            )
        if len(risky) > max_rows:
            lines.append("")
            lines.append(f"... {len(risky) - max_rows} more at-risk campaigns not shown.")
    return "\n".join(lines)


def send_email_alert(body: str, subject: Optional[str] = None) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    recipient = os.getenv("ALERT_TO_EMAIL", "ricky.ferrer@dmagazine.com").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    required = [host, username, password, sender, recipient]
    if not all(required):
        raise RuntimeError("SMTP settings are incomplete. Check .env values.")

    msg = MIMEText(body)
    msg["Subject"] = subject or f"[D Magazine] Campaign Pacing Alert {date.today().isoformat()}"
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP(host=host, port=port, timeout=30) as client:
        if use_tls:
            client.starttls()
        client.login(username, password)
        client.sendmail(sender, [recipient], msg.as_string())


def main() -> None:
    # Keep CI/host environment variables authoritative (for example
    # GOOGLE_APPLICATION_CREDENTIALS in GitHub Actions).
    load_dotenv(override=False)
    args = parse_args()
    columns = load_column_map()
    if args.source == "csv":
        metrics = load_csv_metrics(args.csv, columns)
    else:
        metrics = load_gam_metrics(args.gam_report_id, columns)

    risk_counts = {}
    for m in metrics:
        risk_counts[m.risk_level] = risk_counts.get(m.risk_level, 0) + 1

    source_label = "CSV" if args.source == "csv" else "GAM"
    print(f"Loaded {len(metrics)} campaigns from {source_label}")
    print(f"Risk distribution: {risk_counts}")

    body = format_alert_body(metrics)
    print("\nAlert preview:\n")
    print(body)

    if args.dry_run:
        return

    write_snapshot_to_db(metrics)
    if args.skip_email:
        print("\nSnapshot written to DB. Email skipped.")
        return

    send_email_alert(body)
    print("\nSnapshot written to DB and alert email sent.")


if __name__ == "__main__":
    main()
