#!/usr/bin/env python3
"""Offline tests for relatedGuides enrichment in get_all_deadlines_unified().

Covers category defaults (CATEGORY_RELATED_GUIDES), per-entry overrides
(validity-reduction deadlines), the framework-guide fallback for deadlines
tied to a framework with an on-site guide, and JSON serializability of the
payload. No network calls.
"""
import json

import pki_compliance_mcp as pki

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


unified = pki.get_all_deadlines_unified()
by_id = {d["id"]: d for d in unified}


print("== every deadline carries the field ==")
check("all entries have relatedGuides", all("relatedGuides" in d for d in unified))
check("relatedGuides is always a list", all(isinstance(d["relatedGuides"], list) for d in unified))
check(
    "guide entries have title/url/hasVideo",
    all(
        set(g) == {"title", "url", "hasVideo"}
        for d in unified
        for g in d["relatedGuides"]
    ),
)

print("== category defaults ==")
ct = by_id["chrome-ct-prelogging-required"]["relatedGuides"]
check("CT pre-logging gets the CT guide", [g["url"] for g in ct] == ["/guides/certificate-transparency"])
check("CT guide flagged hasVideo", ct[0]["hasVideo"] is True)
eku = by_id["chrome-clientauth-ica"]["relatedGuides"]
check("eku category gets both EKU guides", len(eku) == 2)
check(
    "leaf-sunset override maps root-store entry to EKU guides",
    by_id["chrome-clientauth-leaf-sunset"]["relatedGuides"] == eku,
)
check(
    "unmapped 'certificates' category stays empty",
    by_id["pre-issuance-linting"]["relatedGuides"] == [],
)

print("== per-entry overrides ==")
for did in ("validity-200-days", "validity-100-days", "validity-47-days", "short-lived-cert-threshold-7-days"):
    check(
        f"{did} gets 47-day timeline guide",
        [g["url"] for g in by_id[did]["relatedGuides"]] == ["/guides/47-day-certificate-timeline"],
    )

print("== new category defaults and id overrides ==")
check(
    "governance category gets CPS guide",
    [g["url"] for g in by_id["chrome-cpcps-attestation-required"]["relatedGuides"]] == ["/guides/what-is-a-cps"],
)
check(
    "algorithm-deprecation gets Hash Functions + RSA vs ECC",
    [g["url"] for g in by_id["sc097-sha1-ca-crl-sunset"]["relatedGuides"]]
    == ["/guides/hash-functions", "/guides/rsa-vs-ecc"],
)
check(
    "csc32 override gets Code Signing",
    [g["url"] for g in by_id["cabf-csc32-reserved-policy-oid"]["relatedGuides"]] == ["/guides/code-signing"],
)
check(
    "mozilla mrsp entry gets MRSP + CPS guides (override beats governance default)",
    [g["url"] for g in by_id["mozilla-mrsp-3-1-effective"]["relatedGuides"]]
    == ["/compliance/mozilla-root-store-policy-v3-1", "/guides/what-is-a-cps"],
)
check(
    "framework deadline id override works (nspm-12 -> CNSA guide)",
    [g["url"] for g in by_id["nspm-12-cnss-governance"]["relatedGuides"]]
    == ["/guides/cnsa-2-certificate-management"],
)

print("== entrust distrust + cabf overrides (empty 'certificates' -> explicit) ==")
for did in ("chrome-entrust-distrust", "apple-entrust-distrust",
            "mozilla-entrust-distrust", "microsoft-entrust-distrust"):
    check(
        f"{did} gets Root Stores guide",
        [g["url"] for g in by_id[did]["relatedGuides"]] == ["/guides/root-stores"],
    )
check(
    "code-signing-validity-460 gets Code Signing guide",
    [g["url"] for g in by_id["code-signing-validity-460"]["relatedGuides"]] == ["/guides/code-signing"],
)
check(
    "sc092-precert-signing-ca-sunset gets CT guide",
    [g["url"] for g in by_id["sc092-precert-signing-ca-sunset"]["relatedGuides"]] == ["/guides/certificate-transparency"],
)

print("== framework-guide fallback ==")
check(
    "framework deadline inherits framework guide",
    [g["url"] for g in by_id["dora-effective"]["relatedGuides"]] == ["/guides/dora-certificate-management"],
)
check(
    "DEADLINES entry with framework_id inherits framework guide",
    [g["url"] for g in by_id["luxembourg-nis2-in-force"]["relatedGuides"]] == ["/guides/nis2-certificate-management"],
)
check(
    "external resource_link (cabforum.org) lends no guide",
    by_id["pre-issuance-linting"]["framework_id"] == "cabforum"
    and by_id["pre-issuance-linting"]["relatedGuides"] == [],
)

print("== serialization ==")
try:
    json.dumps(unified)
    check("unified list JSON-serializable", True)
except (TypeError, ValueError):
    check("unified list JSON-serializable", False)


print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
