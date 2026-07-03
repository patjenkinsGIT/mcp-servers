# PKI Compliance MCP

MCP server + web API that powers the [FixMyCert Compliance Hub](https://fixmycert.com/compliance). Tracks PKI compliance deadlines from CA/Browser Forum ballots, browser root programs (Chrome, Mozilla, Apple, Microsoft), NIST/NSA publications, and regulatory frameworks (DORA, NIS2, UK CSR Bill), with a daily AI research pipeline that proposes updates for human review.

Runs on the DigitalOcean droplet at `/opt/mcp-servers/pki-compliance-mcp`, served as the systemd unit `pki-compliance-api` behind nginx at `https://compliance-api.fixmycert.com`.

## Components

| File | Purpose |
|------|---------|
| `pki_compliance_mcp.py` | MCP server + HTTP API. All compliance data lives inline here (no database): `DEADLINES`, `REGULATORY_FRAMEWORKS`, `CABF_DOCUMENTS`, root stores, CA chains, etc. |
| `compliance_auto_refresh.py` | Daily research cron: cost gate → Claude research with web search → diff analysis → dedup → pending-updates file for review. |
| `daily_doc_check.sh` | Daily document hash check (refreshes document version state in the MCP container). |
| `daily_email.py` | Morning summary email via Resend. |
| `deploy.sh` | Pull + pip install + restart the systemd service (run on the droplet). |
| `test_gating_dedup.py` | Offline tests for the cost gate and dedup logic. |
| `test_source_url_and_estimates.py` | Offline tests for API serialization (`source_url`, `is_estimated`), status/date consistency, and prompt guarantees. |

## Data model

Each `DEADLINES` entry:

```python
{
    "id": "kebab-case-id",           # unique, stable
    "date": "YYYY-MM-DD",            # always full date format
    "title": "Short Title",
    "description": "Full description",
    "source": "cab-forum|chrome|mozilla|apple|microsoft|nist|nsa",
    "category": "certificates|validation|revocation|...",
    "isMajor": True/False,
    # Optional keys:
    "impact": "Brief impact statement",
    "is_estimated": True,            # date is not day-precise (see conventions below)
    "source_url": "https://...",     # link to the authoritative source
    "ballotStatus": "passed",        # tracks the BALLOT, not the deadline
}
```

`REGULATORY_FRAMEWORKS` entries hold their own `deadlines` lists with the same optional keys.

### Conventions (enforced by the research prompt and tests)

- **Date precision** — a day-precise date is only used when a primary source explicitly states that exact day. If a source gives only a month/quarter/year, the entry uses the *last day* of that period and sets `"is_estimated": True`. The front-end renders this as a `~ Est.` badge. Never invent a specific day.
- **`source_url`** — link to the authoritative source (the CABF ballot page, root program policy, EUR-Lex text, official government page). Verify the URL resolves before committing. The front-end renders it as a "Source" link; entries without one render normally. Only add URLs you have actually verified — no guessing (that's the whole point).
- **Status is computed, never stored** — `get_all_deadlines_unified()` derives `status` from the date on every request (`upcoming`/`passed`). Do **not** hardcode `"status"` on deadline entries; it will drift as dates pass. The only exception is `"status": "ongoing"`, which is preserved. `ballotStatus: "passed"` means the *ballot* passed voting — the deadline itself stays `upcoming` until its effective date.

## Daily automated flow (times UTC, cron on the droplet)

**10:00 — `compliance_auto_refresh.py`** (6am ET)
1. **Cost gate** (`should_run_research`) — runs a cheap document-hash check in the MCP container (no AI tokens). Skips the day entirely if no tracked document changed and research ran within the last 7 days. A weekly safety net forces a run; `--force` overrides manually. A skipped day costs zero Anthropic API calls.
2. **Research** — runs the `RESEARCH_QUERIES` via Claude with web search.
3. **Diff analysis** — compares findings to live `/api/compliance-data` under `DIFF_SYSTEM_PROMPT` (date precision rules, mandatory `source_url`, unsure → `needs_human_review`).
4. **Dedup** (`dedup_changes`) — drops proposals already in `DEADLINES` (by id or normalized title+date signature), already-current document versions, and previously rejected ids.
5. **Output** — writes `~/.pki-compliance-mcp/pending_updates_<date>.json`. **Live data is never changed automatically.** (A Sunday `--auto-apply` cron line exists in the crontab but is commented out.)

**10:30 — `daily_doc_check.sh`** refreshes document version state.
**10:35 — `daily_email.py`** sends the morning summary (Resend).

## Reviewing proposals (the only routine human task)

When the morning email / pending-updates file proposes changes:

**Accept a proposed deadline**
1. Verify the proposal's `source_url` actually supports the date (open the link).
2. Add the entry to `DEADLINES` in `pki_compliance_mcp.py`, following the data-model conventions above.
3. Run the tests, commit, push, deploy (see below).

**Reject a proposal** (on the droplet — persists forever, it will never be re-proposed):
```bash
cd /opt/mcp-servers/pki-compliance-mcp
python3 compliance_auto_refresh.py --reject <id> [<id> ...]
python3 compliance_auto_refresh.py --list-rejected   # inspect the rejected list
```

**After a substantive review, bump the freshness stamp** — `DATA_FRESHNESS["lastFullReview"]` (and the relevant `fieldVerifications` dates + `COMPLIANCE_METADATA["lastUpdated"]`) in `pki_compliance_mcp.py`. Nothing bumps these automatically; 45 days after `lastFullReview` the dashboard and the public site show a "data is stale" banner. The ops dashboard lives at `/dashboard?token=<DASHBOARD_TOKEN>` (token in the droplet's `.env`/crontab).

**Manual research runs**
```bash
python3 compliance_auto_refresh.py --force        # bypass the cost gate
python3 compliance_auto_refresh.py --dry-run      # print results, write nothing
python3 compliance_auto_refresh.py --query-only   # research queries only, no diff
```

## Tests

No pytest needed — plain scripts, no network calls:

```bash
python3 -m py_compile pki_compliance_mcp.py compliance_auto_refresh.py
python3 test_gating_dedup.py
python3 test_source_url_and_estimates.py
```

Requires `httpx`, `pydantic`, `feedparser` importable (the `mcp` package is optional). Both suites must pass before deploying. `test_source_url_and_estimates.py` also catches stale hardcoded statuses and regressions in the research-prompt guarantees.

## Deployment

```bash
# from your machine
git push origin main
ssh root@<droplet> "/opt/mcp-servers/pki-compliance-mcp/deploy.sh"
# or manually:
ssh root@<droplet> "cd /opt/mcp-servers && git pull origin main && systemctl restart pki-compliance-api"
```

Verify after deploying:
```bash
curl -s https://compliance-api.fixmycert.com/api/compliance-data | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(len(d['deadlines']), 'deadlines, fetched', d['metadata']['fetchedAt'])"
```

Note: `/api/compliance-data` is served with a 1-hour cache header — browsers may show stale data for up to an hour after a deploy (hard refresh to bypass).

## Front-end (separate codebase)

The Compliance Hub UI lives in the **FixMyCert** Replit app — not in this repo. It fetches `/api/compliance-data` and renders:
- `is_estimated` → `~ Est.` badge
- `source_url` → "Source" link (opens in new tab; absent/null renders nothing)
- "Show Past" toggle → full multi-year past history grouped by year (default view shows last 90 days)

Data-only changes here flow to the site automatically (after cache expiry). Front-end *code* changes are made in Replit and require a republish there.

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/compliance-data` | Everything in one call (front-end) — deadlines, documents, root stores, frameworks |
| `/deadlines`, `/api/compliance/deadlines` | Filterable deadlines (category, framework, jurisdiction, status, within_days) |
| `/api/compliance/upcoming` | Upcoming deadlines within N days |
| `/frameworks`, `/api/compliance/frameworks` | Regulatory framework list |
| `/status`, `/feeds`, `/documents` | Monitor status, feed checks, document versions |
| `/api/news`, `/api/news/sources`, `/api/news/refresh` | News feed aggregation |

MCP tools (`pki_get_deadlines`, `pki_check_all_documents`, etc.) expose the same data over SSE for Claude clients.
