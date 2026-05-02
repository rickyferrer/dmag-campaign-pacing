"""
Incremental WordPress post ingestion + Claude classification.

Usage:
  python src/ingest_posts.py                # fetch posts newer than last DB entry
  python src/ingest_posts.py --since 2026-04-01   # explicit start date
  python src/ingest_posts.py --dry-run      # classify but don't write to DB
  python src/ingest_posts.py --limit 50     # cap posts fetched (useful for testing)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

WP_BASE    = "https://dev.dmagazine.com/wp-json/wp/v2"
WP_FIELDS  = "id,date,slug,title,excerpt,link,type,categories,tags,section"
WP_HEADERS = {"User-Agent": "D Magazine Content Ingest/1.0"}
BATCH_SIZE = 10   # posts per Claude call
DELAY_MS   = 0.3  # seconds between WP pages

DATABASE_URL  = os.environ.get("DATABASE_URL", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

USER_NEEDS_CONTEXT = """
You are classifying news articles using the User Needs Model 2.0 developed by Dmitry Shishkin.
Assign EXACTLY ONE primary user need from this list to each article, based on its title and excerpt.

USER NEEDS:
1. update_me        — Breaking news, what happened, factual updates, announcements, results
2. educate_me       — Explainers, how-things-work, backgrounders, deep dives, context-building
3. give_me_perspective — Analysis, opinion, expert commentary, what-it-means pieces
4. divert_me        — Entertainment, lifestyle, fun, lighter fare, things to do, culture
5. inspire_me       — Profiles of achievement, solution journalism, feel-good, human interest
6. help_me          — Service journalism, guides, how-to, recommendations, directories, tools
7. connect_me       — Community identity, shared experience, belonging, civic pride
8. keep_me_engaged  — Conversation starters, trending topics, reader participation, debate

CLASSIFICATION RULES:
- If a piece is a ranked list / guide (50 Best, Top 10) → help_me
- If a piece announces a personnel move, award, or event result → update_me
- If a piece is a restaurant/arts/culture review with recommendations → divert_me
- If a piece profiles a person overcoming odds or achieving something → inspire_me
- If a piece explains WHY something is happening or provides background → educate_me
- If a piece is a political/business analysis or columnist take → give_me_perspective
- If a piece is explicitly a "things to do" or event calendar → help_me
- "Leading Off" daily news digests → update_me
- Obituaries → inspire_me (unless pure announcement → update_me)

Respond ONLY with a JSON array, no markdown, no explanation. Each element:
{ "id": <post_id>, "user_need": "<need_slug>", "confidence": "high|medium|low", "reason": "<10 words max>" }
""".strip()

# ── WordPress helpers ─────────────────────────────────────────────────────────

def _strip(html: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", "", html)
    return text.replace("&amp;", "&").replace("&#8217;", "'").strip()


def wp_fetch_page(after: str, page: int) -> tuple[list[dict], int, int]:
    url = (
        f"{WP_BASE}/posts"
        f"?after={after}&per_page=100&page={page}"
        f"&_fields={WP_FIELDS}&status=publish"
    )
    req = urllib.request.Request(url, headers=WP_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        total_posts = int(resp.headers.get("X-WP-Total", 0))
        posts = json.loads(resp.read())
    return posts, total_pages, total_posts


def fetch_posts(after: str, limit: int | None = None) -> list[dict]:
    print(f"\n── Fetching WP posts after {after} ──")
    raw, total_pages, total_posts = wp_fetch_page(after, 1)
    print(f"  Found {total_posts} posts across {total_pages} pages")
    all_posts = list(raw)

    for page in range(2, total_pages + 1):
        if limit and len(all_posts) >= limit:
            break
        print(f"  Page {page}/{total_pages}...", end="\r")
        raw, _, _ = wp_fetch_page(after, page)
        all_posts.extend(raw)
        time.sleep(DELAY_MS)

    print()
    if limit:
        all_posts = all_posts[:limit]

    return [
        {
            "post_id":   p["id"],
            "date":      p["date"],
            "slug":      p["slug"],
            "title":     _strip(p.get("title", {}).get("rendered", "")),
            "excerpt":   _strip(p.get("excerpt", {}).get("rendered", "")),
            "link":      p.get("link", ""),
            "post_type": p.get("type", ""),
            "categories": "|".join(str(c) for c in (p.get("categories") or [])),
            "tags":       "|".join(str(t) for t in (p.get("tags") or [])),
            "section":    "|".join(str(s) for s in (p.get("section") or [])),
        }
        for p in all_posts
    ]

# ── Classification ────────────────────────────────────────────────────────────

def classify_batch(client: anthropic.Anthropic, articles: list[dict]) -> list[dict]:
    items = [
        {"id": a["post_id"], "title": a["title"], "excerpt": a["excerpt"][:300]}
        for a in articles
    ]
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": USER_NEEDS_CONTEXT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"ARTICLES TO CLASSIFY:\n{json.dumps(items, ensure_ascii=False)}",
            }
        ],
    )
    raw = next((b.text for b in resp.content if b.type == "text"), "[]")
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def classify_all(client: anthropic.Anthropic, posts: list[dict]) -> dict[int, dict]:
    print(f"\n── Classifying {len(posts)} posts with Claude ──")
    results: dict[int, dict] = {}
    batches = [posts[i:i + BATCH_SIZE] for i in range(0, len(posts), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        print(f"  Batch {i + 1}/{len(batches)} ({len(batch)} articles)...", end="\r")
        try:
            classified = classify_batch(client, batch)
            for c in classified:
                results[c["id"]] = c
        except Exception as exc:
            print(f"\n  Batch {i + 1} failed: {exc} — marking unclassified")
            for a in batch:
                results[a["post_id"]] = {
                    "id": a["post_id"],
                    "user_need": "unclassified",
                    "confidence": "low",
                    "reason": "api error",
                }
        time.sleep(0.5)

    print(f"\n  Classified: {len(results)} articles")
    return results

# ── Database ──────────────────────────────────────────────────────────────────

def get_last_ingested_date(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM classified_posts")
        row = cur.fetchone()
    if row and row[0]:
        return row[0].isoformat()
    # Default: 24 hours ago
    return (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()


def upsert_posts(conn, posts: list[dict]) -> int:
    sql = """
        INSERT INTO classified_posts
            (post_id, date, slug, title, excerpt, link, post_type,
             categories, tags, section, user_need, confidence, un_reason, classified_at)
        VALUES
            (%(post_id)s, %(date)s, %(slug)s, %(title)s, %(excerpt)s, %(link)s,
             %(post_type)s, %(categories)s, %(tags)s, %(section)s,
             %(user_need)s, %(confidence)s, %(un_reason)s, NOW())
        ON CONFLICT (post_id) DO UPDATE SET
            date         = EXCLUDED.date,
            slug         = EXCLUDED.slug,
            title        = EXCLUDED.title,
            excerpt      = EXCLUDED.excerpt,
            link         = EXCLUDED.link,
            post_type    = EXCLUDED.post_type,
            categories   = EXCLUDED.categories,
            tags         = EXCLUDED.tags,
            section      = EXCLUDED.section,
            user_need    = EXCLUDED.user_need,
            confidence   = EXCLUDED.confidence,
            un_reason    = EXCLUDED.un_reason,
            classified_at = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, posts)
    conn.commit()
    return len(posts)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="ISO date to fetch posts after (default: last DB entry)")
    parser.add_argument("--dry-run", action="store_true", help="Classify but skip DB write")
    parser.add_argument("--limit", type=int, help="Max posts to fetch")
    args = parser.parse_args()

    if not ANTHROPIC_KEY:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    if not DATABASE_URL and not args.dry_run:
        sys.exit("ERROR: DATABASE_URL not set")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Determine since date
    if args.since:
        since = args.since
    elif args.dry_run:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    else:
        with psycopg.connect(DATABASE_URL) as conn:
            since = get_last_ingested_date(conn)

    print(f"  Since: {since}")

    posts = fetch_posts(since, limit=args.limit)
    if not posts:
        print("No new posts found.")
        return

    classifications = classify_all(client, posts)

    for post in posts:
        c = classifications.get(post["post_id"], {})
        post["user_need"]  = c.get("user_need", "unclassified")
        post["confidence"] = c.get("confidence", "low")
        post["un_reason"]  = c.get("reason", "")

    # Distribution summary
    counts: dict[str, int] = {}
    for p in posts:
        counts[p["user_need"]] = counts.get(p["user_need"], 0) + 1
    print("\n── Distribution ──")
    for need, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / len(posts) * 100
        bar = "█" * round(pct / 3)
        print(f"  {need:<22} {count:>4}  {pct:>5.1f}%  {bar}")

    if args.dry_run:
        print("\nDry run — skipping DB write.")
        return

    with psycopg.connect(DATABASE_URL) as conn:
        written = upsert_posts(conn, posts)
    print(f"\n  Written to DB: {written} posts\nDone.")


if __name__ == "__main__":
    main()
