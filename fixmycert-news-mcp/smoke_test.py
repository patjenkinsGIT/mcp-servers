"""
End-to-end smoke test for fixmycert-news-mcp against the live news API.

Run inside the deployed container (env vars already set there):

    docker compose exec fixmycert-news python smoke_test.py

Flow: news_add draft -> visible via admin, NOT on public /api/news ->
news_publish -> visible on public /api/news -> news_dashboard +
news_get_uncovered sanity -> news_archive cleanup (no hard delete exists).
Does not touch the aggregator, cron, or /compliance.
"""

import asyncio
import re
import sys

import httpx

import server as s


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    return ok


async def main() -> int:
    failures = 0
    test_title = "SMOKE TEST fixmycert-news-mcp (safe to archive)"

    internal_url_v1 = "/guides/smoke-test-internal-link"
    internal_url_v2 = "/guides/smoke-test-internal-link-updated"

    # 1. Create draft (with internalUrl to prove POST accepts it)
    out = await s.news_add(s.AddNewsInput(
        title=test_title,
        excerpt="Automated smoke test item from fixmycert-news-mcp deploy verification.",
        newsCluster=s.NewsCluster.OTHER,
        internalUrl=internal_url_v1,
    ))
    m = re.search(r"\*\*id\*\*: ([0-9a-f-]{8,})", out)
    if not check("news_add created draft", bool(m), out[:200] if not m else m.group(1)):
        return 1
    item_id = m.group(1)

    try:
        # 2. Draft shows via admin list
        admin_out = await s.news_list(s.ListNewsInput(status=s.Status.DRAFT, isFirstParty=True, limit=200))
        failures += not check("draft visible via news_list (admin)", item_id in admin_out or test_title in admin_out)

        # 2b. internalUrl persisted on create (news_get reads it back from the API)
        got = await s.news_get(s.GetNewsInput(id=item_id))
        failures += not check("internalUrl persisted by POST", internal_url_v1 in got, got[:200])

        # 2c. internalUrl updatable via PATCH
        out = await s.news_update(s.UpdateNewsInput(id=item_id, internalUrl=internal_url_v2))
        failures += not check("news_update internalUrl", "updated" in out.lower(), out[:120])
        got = await s.news_get(s.GetNewsInput(id=item_id))
        failures += not check("internalUrl persisted by PATCH", internal_url_v2 in got, got[:200])

        # 3. Draft NOT on public feed
        async with httpx.AsyncClient(timeout=30) as c:
            pub = (await c.get(s.NEWS_API_BASE_URL + "/api/news", params={"limit": 200})).json()
        pub_ids = {i.get("id") for i in pub.get("items", [])}
        failures += not check("draft NOT on public /api/news", item_id not in pub_ids)

        # 4. Publish -> appears on public feed
        out = await s.news_publish(s.ItemIdInput(id=item_id))
        failures += not check("news_publish", "Published" in out, out[:120])
        async with httpx.AsyncClient(timeout=30) as c:
            pub = (await c.get(s.NEWS_API_BASE_URL + "/api/news", params={"limit": 200})).json()
        pub_ids = {i.get("id") for i in pub.get("items", [])}
        failures += not check("published item on public /api/news", item_id in pub_ids)

        # 5. Dashboard + uncovered sanity
        dash = await s.news_dashboard(s.DashboardInput(response_format=s.ResponseFormat.JSON))
        failures += not check("news_dashboard sane", '"total"' in dash and '"byCluster"' in dash, dash[:120])
        unc = await s.news_get_uncovered(s.GetUncoveredInput())
        failures += not check("news_get_uncovered sane", not unc.startswith("❌"), unc[:120])
    finally:
        # 6. Cleanup: archive the test item (removes from public feed)
        out = await s.news_archive(s.ItemIdInput(id=item_id))
        failures += not check("news_archive cleanup", "Archived" in out, out[:120])

    print("\n" + ("SMOKE TEST PASSED" if failures == 0 else f"SMOKE TEST: {failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
