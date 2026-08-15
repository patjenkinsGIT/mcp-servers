"""
PKI Compliance Monitor MCP Server

An MCP server for monitoring PKI compliance updates from authoritative sources:
- CA/Browser Forum ballots and requirements
- Chrome Root Program policy changes
- Mozilla Root Store Policy updates
- Apple/Microsoft root program changes
- NIST publications
- CCADB discussions

Usage:
    python pki_compliance_mcp.py                    # stdio transport (local)
    python pki_compliance_mcp.py --http --port 8000 # HTTP transport (remote)
"""

import json
import hashlib
import re
import csv
import io
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict

import httpx
import feedparser
from pydantic import BaseModel, Field, ConfigDict

# MCP SDK is optional - only needed for --http mode (not --simple-http)
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

# ============================================================================
# Configuration
# ============================================================================

import os as _os
DATA_DIR = Path(_os.environ.get("DATA_DIR", str(Path.home() / ".pki-compliance-mcp")))
STATE_FILE = DATA_DIR / "state.json"
CACHE_FILE = DATA_DIR / "cache.json"

# Feed and document sources
FEEDS = {
    "cabforum_public": {
        "name": "CA/Browser Forum Public List",
        # The old pipermail archive (lists.cabforum.org) is gone, not merely frozen:
        # DNS resolves but 80/443 time out and ICMP is dropped (checked 2026-08-05 from
        # two networks). Current traffic and the full public archive live on Google Groups,
        # which is readable anonymously.
        "url": "https://groups.google.com/a/groups.cabforum.org/g/public",
        "type": "google_group",
        "priority": "high",
    },
    "ccadb_public": {
        "name": "CCADB Public Discussions",
        "url": "https://groups.google.com/a/ccadb.org/g/public",
        "type": "google_group",
        "priority": "high",
    },
    "chrome_security_blog": {
        "name": "Google Security Blog",
        "url": "https://security.googleblog.com/feeds/posts/default",
        "type": "atom",
        "priority": "high",
    },
    "mozilla_security": {
        "name": "Mozilla Security Blog",
        "url": "https://blog.mozilla.org/security/feed/",
        "type": "rss",
        "priority": "medium",
    },
    "microsoft_root_program": {
        # Official source of truth since Oct 2025 (repo README supersedes the
        # learn.microsoft.com pages, which now carry a "superceded" notice).
        # GitHub's commits.atom covers every requirement/release-note change,
        # replacing the old high-priority manual monthly check.
        "name": "Microsoft Trusted Root Program (GitHub commits)",
        "url": "https://github.com/TrustedRootProgram/Program-Requirements/commits.atom",
        "type": "atom",
        "priority": "high",
    },
    "apple_root_program": {
        # Same shape as the Microsoft entry: the repo is Apple's authoritative
        # publication point (its README says so; repo created 2026-02-19), and
        # the DOCUMENTS hash-watch on raw policy.md can only say "the hash
        # moved". commits.atom attributes each move to a dated commit and
        # surfaces changes before they land in policy.md.
        "name": "Apple Root Program (GitHub commits)",
        "url": "https://github.com/apple/apple-root-program/commits.atom",
        "type": "atom",
        "priority": "high",
    },
}

DOCUMENTS = {
    "cabf_br": {
        "name": "CA/Browser Forum Baseline Requirements",
        "url": "https://cabforum.org/baseline-requirements-documents/",
        "check_url": "https://cabforum.org/working-groups/server/baseline-requirements/documents/",
        "priority": "high",
    },
    "chrome_root_policy": {
        "name": "Chrome Root Program Policy",
        "url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "priority": "high",
    },
    "mozilla_root_policy": {
        # Track the canonical policy source, not the wiki landing page — the
        # wiki page carries no version text and its hash never changed when
        # MRSP v3.1 shipped (missed on 2026-07-01).
        "name": "Mozilla Root Store Policy",
        "url": "https://raw.githubusercontent.com/mozilla/pkipolicy/master/rootstore/policy.md",
        "priority": "high",
    },
    "apple_root_program": {
        "name": "Apple Root Certificate Program",
        # Authoritative publication point moved to GitHub (repo created 2026-02-19;
        # README calls it authoritative). Old apple.com page still serves the policy
        # but no longer canonical. Hash the raw markdown — stable, no page chrome.
        "url": "https://github.com/apple/apple-root-program",
        "check_url": "https://raw.githubusercontent.com/apple/apple-root-program/main/policy.md",
        "priority": "medium",
    },
    "microsoft_root_program": {
        # Repointed 2026-07-21: the docs.microsoft.com page was superseded in
        # Oct 2025 by the GitHub repo and went stale (we watched a dead page
        # for ~9 months). Primary watch is now the commits.atom feed in FEEDS;
        # this hash of the raw requirements markdown is a backstop — raw
        # markdown has no dynamic page chrome, so hashing is reliable.
        "name": "Microsoft Trusted Root Program",
        "url": "https://github.com/TrustedRootProgram/Program-Requirements",
        "check_url": "https://raw.githubusercontent.com/TrustedRootProgram/Program-Requirements/main/Requirements.md",
        "priority": "medium",
    },
    "microsoft_root_announcements": {
        # Added 2026-07-30. Microsoft created Announcements.md on 2026-07-29 as
        # the official channel for Trusted Root Program announcements; we only
        # hashed Requirements.md, so program announcements were invisible to the
        # doc check. Separate doc_id (not a second URL on microsoft_root_program)
        # because document_hashes is keyed by doc_id — one hash per file, so an
        # announcement is distinguishable from a requirements change.
        # NOTE: the file lives at the repo ROOT, not under trusted-root/.
        "name": "Microsoft Trusted Root Program Announcements",
        "url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/Announcements.md",
        "check_url": "https://raw.githubusercontent.com/TrustedRootProgram/Program-Requirements/main/Announcements.md",
        "priority": "high",
    },
    "nist_800_131a": {
        "name": "NIST SP 800-131A (Algorithm Transitions)",
        "url": "https://csrc.nist.gov/publications/detail/sp/800-131a/rev-2/final",
        "priority": "high",
    },
    "microsoft_release_notes": {
        # Added 2026-08-07. Microsoft's monthly DEPLOYMENT NOTICES land at
        # trusted-root/YYYY/<month>-YYYY.md — a path that churns every month and
        # so cannot be a stable tracked URL. Release Notes.md is the repo-root
        # INDEX that gets one row per notice (back to 2019), so it changes
        # whenever a notice is published while keeping a fixed path.
        # Why this exists: the August 2026 notice (release 2026-08-25, NotBefore
        # 2026-09-15 — the largest root-store change tracked here) reached us
        # only via the GitHub-commits feed. No tracked document covered it —
        # microsoft_root_program hashes Requirements.md and
        # microsoft_root_announcements hashes Announcements.md. This index was
        # committed 2026-08-05T18:37:16Z, 19 minutes before that Announcements
        # change, so tracking it would have caught the notice on its own.
        # NOTE the space in the filename must stay percent-encoded as %20.
        "name": "Microsoft Trusted Root Program Release Notes",
        "url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/Release%20Notes.md",
        "check_url": "https://raw.githubusercontent.com/TrustedRootProgram/Program-Requirements/main/Release%20Notes.md",
        "priority": "high",
    },
    "nist_800_57": {
        "name": "NIST SP 800-57 (Key Management)",
        "url": "https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final",
        "priority": "medium",
    },
}

# Sources that don't have RSS feeds and require manual checking
# The /feeds endpoint will remind users to check these manually.
# Empty since 2026-07-21: the Microsoft release-notes entry (last holdout)
# moved to FEEDS as the Program-Requirements repo commits.atom feed.
MANUAL_CHECK_REQUIRED = []

# =============================================================================
# COMPLIANCE DATA - This is the single source of truth
# Update here, frontend fetches automatically
# =============================================================================

# =============================================================================
# BALLOTS TO WATCH - Update DEADLINES when these pass
# =============================================================================
#
# SC097: SHA-1 Signature Sunset (CA Certificates & CRLs)
#   Status: PASSED (Jan 24, 2026) - Entered IPR Review Period
#   Voting: 26 YES / 0 NO (Issuers), 4 YES / 0 NO (Consumers) - Unanimous
#   Enforcement: September 15, 2026
#   What: Revoke all SHA-1 signed CA certs, migrate CRL signing to SHA-256+
#   Source: https://github.com/cabforum/servercert/pull/645
#   Status: EFFECTIVE (Feb 25, 2026) - IPR review complete
#   No further action needed
#
# SC095: Clean-up 2025
#   Status: Discussion period
#   What: Various clarifications and cleanup (non-normative)
#   Action: No deadline entry needed unless normative changes added
#
# SC087: Registration Number Improvement for EV Certificates
#   Status: Draft (expected Jan 2025)
#   What: EV certificate registration number requirements
#   Action: Review for deadline implications when discussion starts
#
# -----------------------------------------------------------------------------
# Monthly review checklist:
# - [ ] Check https://cabforum.org for ballot status changes
# - [ ] Check servercert-wg mailing list digest for new ballots
# - [ ] Update any "proposed" entries that have passed voting + IPR
# =============================================================================

# Each entry: id, date, title, description, source, category, isMajor.
# Optional keys: impact, is_estimated (date is not day-precise, rendered as
# "~ Est." badge), source_url (link to the authoritative source).
#
# Status is computed from the date — never hardcode it — with one exception:
# set "status": "ongoing" explicitly on type (b) entries.
#   type (b) = the date STARTED a rule whose failure mode recurs per-reader
#              after that date (renewal traps like NotBefore distrusts,
#              rolling processes, in-force regimes) -> "ongoing", rendered
#              as an amber "In force" badge, never green "Completed".
#   type (a) = the ecosystem transitioned once and the rule is now baseline
#              (requirement effective dates, due dates, one-time events)
#              -> no status key; computes "passed" after the date.
# The distinction is editorial and cannot be inferred from category — decide
# it per entry when adding one. Applies identically to framework sub-list
# deadlines. Front-end mapping: FRONTEND_ONGOING_STATUS_SPEC.md. Tests lock
# the classification in test_source_url_and_estimates.py.
DEADLINES = [
    {
        "id": "chrome-digicert-legacy-roots-distrust",
        "date": "2026-07-01",
        "status": "ongoing",
        "title": "Chrome distrust: DigiCert Trusted Root G4 / Assured ID G2/G3 (new issuance)",
        "description": "Chrome Root Program removes DigiCert Trusted Root G4, Assured ID G2, and Assured ID G3 (not dedicated TLS hierarchies — they also issued Code Signing/Timestamp ICAs). TLS certificates issued on or after 2026-07-01 from these roots are not Chrome-trusted; certificates issued before remain trusted until expiry. New issuance and reissues must chain to DigiCert Global Root G2 (RSA) or G3 (ECC).",
        "source": "chrome",
        "source_url": "https://knowledge.digicert.com/alerts/google-chrome-root-removal-trusted-root-g4-assured-id-g2-id-g3",
        "category": "root-store",
        "isMajor": True,
        "impact": "DigiCert customers on legacy roots must ensure new and reissued TLS certificates chain to Global Root G2 (RSA) or G3 (ECC) to remain Chrome-trusted.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "Chrome removes DigiCert Trusted Root G4, Assured ID G2 and Assured ID G3 from its trust store: TLS certificates issued on or after 2026-07-01 from these hierarchies are not Chrome-trusted, while certificates issued before remain trusted until they expire. New and reissued certificates must chain to DigiCert Global Root G2 (RSA) or G3 (ECC).",
            "scenario": "The break shows up at reissue, not on the deadline date. A DigiCert customer reissues in August 2026, the certificate is perfectly valid, monitoring stays green, and Chrome throws a full-page interstitial because the chain still climbs to Assured ID G2. Anything holding the legacy roots — load balancer chain files, appliance trust stores, pinned mobile clients — needs the Global Root G2/G3 chain staged before its first post-July reissue.",
        },
    },
    {
        "id": "mozilla-dcr-audit-periods",
        "date": "2027-07-01",
        "title": "Mozilla DCRs required — audit periods starting on/after 2027-07-01",
        "description": "Per MRSP v3.1, CA operators with TLS-enabled roots must obtain Detailed Controls Reports (DCRs) for audit periods beginning on or after July 1, 2027, giving Mozilla and auditors visibility into control design and operating effectiveness.",
        "source": "mozilla",
        "source_url": "https://blog.mozilla.org/security/2026/06/29/improving-transparency-and-assurance-in-the-web-pki-mozilla-root-store-policy-v3-1/",
        "category": "audit",
        "isMajor": True,
        "impact": "CAs with TLS-enabled roots in the Mozilla program must engage auditors for Detailed Controls Reports covering audit periods starting on or after 2027-07-01.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "CAs with TLS-enabled roots in the Mozilla program must obtain Detailed Controls Reports for audit periods starting on or after this date; failure risks root-program action.",
            "scenario": "No action for most organizations unless you operate a publicly-trusted CA. Indirect benefit: more auditor visibility into the CAs you rely on.",
        },
    },
    {
        "id": "chrome-cpcps-attestation-required",
        "date": "2026-06-15",
        "title": "Chrome CP/CPS Policy Adherence Attestation Required",
        "description": "Chrome Root Program v1.8: a participant's CP or combined CP/CPS must explicitly state adherence to the latest published version of the Chrome Root Program Policy and the CCADB Policy.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "governance",
        "isMajor": False,
        "impact": "Chrome-trusted CAs must update CP/CPS statements to explicitly attest policy compliance.",
    },
    {
        "id": "eo-14412-hva-pqc-key-establishment",
        "date": "2030-12-31",
        "title": "EO 14412: Federal HVAs must use PQC for key establishment",
        "description": "Executive Order 14412 (June 22, 2026) requires federal agencies to transition all High Value Assets and high impact systems to NIST-approved PQC FIPS for key establishment (e.g., ML-KEM in TLS/IPsec) by December 31, 2030. OMB M-26-15 (June 24, 2026) operationalizes this for civilian agencies.",
        "source": "nist",
        "source_url": "https://www.whitehouse.gov/presidential-actions/2026/06/securing-the-nation-against-advanced-cryptographic-attacks/",
        "category": "pqc",
        "isMajor": True,
        "impact": "Federal HVAs and high-impact systems must adopt ML-KEM key establishment by end of 2030; contractors must comply with PQC FIPS.",
    },
    {
        "id": "eo-14412-hva-pqc-digital-signatures",
        "date": "2031-12-31",
        "title": "EO 14412: Federal HVAs must use PQC for digital signatures",
        "description": "Executive Order 14412 (June 22, 2026) requires federal agencies to transition all High Value Assets and high impact systems to NIST-approved PQC digital signatures (e.g., ML-DSA/SLH-DSA in certificates and code signing) by December 31, 2031.",
        "source": "nist",
        "source_url": "https://www.whitehouse.gov/presidential-actions/2026/06/securing-the-nation-against-advanced-cryptographic-attacks/",
        "category": "pqc",
        "isMajor": True,
        "impact": "Federal certificate and code-signing infrastructure must move to PQC signatures by end of 2031.",
    },
    {
        "id": "dow-pqc-strategy-systems-support",
        "date": "2030-12-31",
        "title": "DoW PQC Strategy: all defense systems must support PQC",
        "description": "The Department of War Post-Quantum Cryptography Strategy (June 23, 2026) requires all DoW systems to support PQC, or be phased out, by December 31, 2030, across NSA-certified High Assurance ECU and NIST-based Commercial Solutions tracks.",
        "source": "nsa",
        "source_url": "https://www.war.gov/News/Releases/Release/Article/4524599/securing-global-dominance-dow-unleashes-quantum-defense-strategy-to-harden-netw/",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "dow-pqc-strategy-full-use",
        "date": "2031-12-31",
        "title": "DoW PQC Strategy: all defense systems must use PQC",
        "description": "The Department of War Post-Quantum Cryptography Strategy (June 23, 2026) requires every DoW system to actively use PQC, unless otherwise specified, no later than December 31, 2031.",
        "source": "nsa",
        "source_url": "https://www.war.gov/News/Releases/Release/Article/4524599/securing-global-dominance-dow-unleashes-quantum-defense-strategy-to-harden-netw/",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "mozilla-mrsp-3-1-effective",
        "date": "2026-07-01",
        "title": "Mozilla Root Store Policy v3.1 Effective",
        "description": "MRSP v3.1 takes effect: root inclusion requests are accepted only for root CA key pairs generated no more than 5 years before submission, and CP/CPS documentation quality requirements are tightened (Section 3.3). Shifts focus to CA documentation transparency and audit reporting.",
        "source": "mozilla",
        "source_url": "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "category": "governance",
        "isMajor": True,
        "impact": "CAs in the Mozilla root program must begin aligning CP/CPS documentation and audit practices with new v3.1 requirements.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "MRSP v3.1 applies from 2026-07-01: Mozilla accepts root inclusion requests only for root CA key pairs generated no more than five years before submission, and Section 3.3's CP/CPS content and quality requirements tighten (full compliance due 2027-07-01).",
            "scenario": "No action unless you operate a CA. The knock-on effect for enterprises is a helpful one: CP/CPS documents move to versioned, publicly hosted, RFC 3647-structured text, which makes them far easier to cite in a vendor review than the PDFs most CAs publish today.",
        },
    },
    {
        "id": "mozilla-cpcps-content-compliance-deadline",
        "date": "2027-07-01",
        "title": "Mozilla CP/CPS Content Requirements Full Compliance Deadline",
        "description": "CAs SHALL comply with MRSP item 2 and Sections 3.3.1 through 3.3.6 (CP/CPS content and quality requirements) no later than July 1, 2027.",
        "source": "mozilla",
        "source_url": "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "category": "governance",
        "isMajor": True,
        "impact": "CAs must finalize updated CP/CPS documentation meeting enhanced content and audit transparency standards.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "CAs must have CP/CPS documentation meeting MRSP v3.1's enhanced content and transparency standards.",
            "scenario": "CA-facing only. Worth noting in vendor reviews; not an enterprise work item.",
        },
    },
    {
        "id": "ocsp-15-min",
        "date": "2025-01-15",
        "title": "OCSP within 15 minutes of issuance",
        "description": "OCSP responses for Subscriber Certificates MUST be available within 15 minutes after certificate issuance.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2024/10/14/ballot-sc076v2-clarify-and-improve-ocsp-requirements/",
        "category": "revocation",
        "isMajor": False,
    },
    {
        "id": "pre-issuance-linting",
        "date": "2025-03-15",
        "title": "Pre-issuance Linting REQUIRED",
        "description": "CAs SHALL implement a linting process to test technical conformity of certificates BEFORE signing.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2024/08/05/ballot-sc075-pre-sign-linting/",
        "category": "certificates",
        "isMajor": True,
        "impact": "CAs must lint all certs before signing",
        "consequences": {
            "enforcement": "SC-075 requires publicly-trusted CAs to run a linting process against the certificate before signing it, so technical non-conformity is caught before issuance rather than after.",
            "scenario": "No action unless you operate a CA. What changes for enterprises is the failure mode: a malformed request that a CA would once have signed now comes back as a rejected order. If an internal tool assembles CSRs by hand — odd SANs, a stray OU, an unqualified name — expect issuance to start failing on requests that used to work, with the rejection arriving from the CA rather than from your own tooling.",
        },
    },
    {
        "id": "multi-perspective-validation",
        "date": "2025-03-15",
        "title": "Multi-Perspective Issuance Corroboration",
        "description": "CAs MUST corroborate domain validation from multiple network perspectives to mitigate BGP hijacking attacks.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2024/08/05/ballot-sc067v3-require-domain-validation-and-caa-checks-to-be-performed-from-multiple-network-perspectives-corroboration/",
        "category": "validation",
        "isMajor": True,
        "impact": "Prevents BGP hijacking attacks",
        "consequences": {
            "enforcement": "SC-067v3 requires CAs to corroborate domain validation and CAA checks from multiple network perspectives before issuance, for the affected methods in TLS BR sections 3.2.2.4 and 3.2.2.5.",
            "scenario": "No action unless you operate a CA, but it changes how validation failures look. The challenge now has to succeed from several vantage points, so split-horizon DNS, geo-fenced WAF rules, or a firewall that only allows the CA's primary validator can fail an order with no error visible on your side. If ACME orders start failing intermittently, check that the challenge is reachable from everywhere — not just from your own network.",
        },
    },
    {
        "id": "chrome-clientauth-ica",
        "date": "2025-06-15",
        "status": "ongoing",
        "title": "Chrome Stops Trusting Mixed-EKU ICAs",
        "description": "Chrome will no longer trust intermediate certificates carrying both ServerAuth and ClientAuth EKUs.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "eku",
        "isMajor": False,
    },
    {
        "id": "digicert-clientauth-default",
        "date": "2025-10-01",
        "title": "DigiCert Removes ClientAuth by Default",
        "description": "DigiCert stops including Client Authentication EKU by default. Must opt-in during enrollment.",
        "source": "chrome",
        "source_url": "https://knowledge.digicert.com/alerts/sunsetting-client-authentication-eku-from-digicert-public-tls-certificates",
        "category": "eku",
        "isMajor": False,
    },
    {
        "id": "sectigo-clientauth-default",
        "date": "2025-10-07",
        "title": "Sectigo Removes ClientAuth by Default",
        "description": "Sectigo stops including Client Authentication EKU by default in newly issued certificates.",
        "source": "chrome",
        "source_url": "https://www.sectigo.com/resource-library/deprecation-of-client-authentication-eku-from-sectigo-ssl-tls-certificates",
        "category": "eku",
        "isMajor": False,
    },
    {
        "id": "mass-revocation-plan",
        "date": "2025-12-01",
        "title": "Mass Revocation Plan REQUIRED",
        "description": "CAs must document mass revocation procedures in CPS.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/07/22/ballot-sc-089-mass-revocation-planning/",
        "category": "revocation",
        "isMajor": False,
    },
    {
        "id": "sc091-persistent-dcv-ip-available",
        "date": "2025-12-16",
        "title": "Persistent DCV TXT for IP Addresses AVAILABLE",
        "description": "SC-091: New method 3.2.2.5.8 ('DNS TXT Record with Persistent Value in Reverse Namespace') now available for IP address validation. Provides DNS-based alternative to deprecated reverse lookup method.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/12/ballot-sc-091-sunset-3.2.2.5.3-reverse-address-lookup-validation-proposal-of-new-dns-based-validation-using-persistent-dcv-txt-record-for-ip-addresses/",
        "category": "validation",
        "isMajor": False,
        "impact": "New validation option for IP address certificates",
    },
    {
        "id": "code-signing-validity-460",
        "date": "2026-03-01",
        "title": "Code Signing MAX VALIDITY: 460 Days",
        "description": "Maximum validity period for Code Signing Certificates reduced from 39 months to 460 days. Most CAs stopped issuing 2-year and 3-year code signing certs in late 2025 to prepare.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/17/ballot-csc-31-maximum-validity-reduction/",
        "category": "certificates",
        "isMajor": True,
        "impact": "All code signing certificate renewals must comply with 460-day maximum",
        "consequences": {
            "enforcement": "Ballot CSC-31 reduces the maximum validity of publicly-trusted code signing certificates from 39 months to 460 days, effective 2026-03-01.",
            "scenario": "Code signing renewal goes from a roughly three-year event to an annual one, and each cycle on an HSM-backed or token-based key means re-attesting, re-keying, and updating every build agent that holds the credential. A pipeline signing with a certificate nobody has rotated since 2023 fails its first release after expiry, usually inside a release window. Already-timestamped signatures keep validating.",
        },
    },
    {
        "id": "validity-200-days",
        "date": "2026-03-15",
        "title": "MAX VALIDITY: 200 DAYS",
        "description": "Maximum certificate validity reduced from 398 to 200 days. Domain validation reuse (DCV) also reduced to 200 days. Subject Identity Information (SII) reuse for OV/EV certificates reduced from 825 to 398 days.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/04/11/ballot-sc081v3-introduce-schedule-of-reducing-validity-and-data-reuse-periods/",
        "category": "certificates",
        "isMajor": True,
        "impact": "Plan automation NOW",
        "consequences": {
            "enforcement": "Per SC-081v3, the TLS Baseline Requirements cap subscriber certificate validity at 200 days for certificates issued from 2026-03-15 to 2027-03-14 and cut domain/IP validation data reuse to 200 days; Subject Identity Information reuse drops to 398 days on the same date.",
            "scenario": "A team that renews annually now renews roughly twice a year, and the DCV reuse cut means the domain-control check can no longer ride on last year's validation. Manual processes built around an annual calendar reminder start missing. This is the step where ACME automation stops being optional for anything beyond a handful of certificates — and it tightens again to 100 days in March 2027 and 47 days in March 2029.",
        },
    },
    {
        "id": "sii-reuse-398-days",
        "date": "2026-03-15",
        "title": "OV/EV Organization Validation Reuse: 398 Days",
        "description": "Subject Identity Information (SII) reuse period drops from 825 days to 398 days for certificates issued on or after March 15, 2026. Affects all OV and EV certificate holders — organizations must redo identity validation more frequently. DV certificates are not affected.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/04/11/ballot-sc081v3-introduce-schedule-of-reducing-validity-and-data-reuse-periods/",
        "category": "validation",
        "isMajor": False,
        "impact": "OV/EV subscribers must redo org validation more frequently; breaks 'set it and forget it' workflows for high-assurance certificates",
    },
    {
        "id": "short-lived-cert-threshold-7-days",
        "date": "2026-03-15",
        "title": "Short-Lived Certificate Threshold: 7 Days",
        "description": "The definition of a Short-lived Subscriber Certificate changes from a validity period of 10 days or less to 7 days or less. Short-lived certificates are exempt from OCSP and CRL requirements. CAs issuing short-lived certs must adjust issuance pipelines to meet the new threshold.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2023/07/14/ballot-sc063v4-make-ocsp-optional-require-crls-and-incentivize-automation/",
        "category": "certificates",
        "isMajor": False,
        "impact": "Affects CAs and subscribers relying on short-lived certificate exemptions from revocation requirements",
    },
    {
        "id": "dnssec-validation",
        "date": "2026-03-15",
        "title": "DNSSEC Validation REQUIRED",
        "description": "DNSSEC validation required for all DCV and CAA lookups.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/06/18/ballot-sc-085v2-require-validation-of-dnssec-when-present-for-caa-and-dcv-lookups/",
        "category": "validation",
        "isMajor": False,
    },
    {
        "id": "sc090-crossover-method-sunset",
        "date": "2026-03-15",
        "title": "IP Address Validation Method SUNSET (Crossover Attacks)",
        "description": "SC-090: Method 3.2.2.4.8 ('IP Address') prohibited. This method enabled crossover attacks via IP reassignment and host header vulnerabilities. Use DNS or HTTP challenges instead.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/20/ballot-sc-090-gradually-sunset-all-remaining-email-based-phone-based-and-crossover-validation-methods-from-sections-3.2.2.4-and-3.2.2.5/",
        "category": "validation",
        "isMajor": False,
        "impact": "Affects rare edge cases where IP lookup was used for domain validation",
    },
    {
        "id": "sc092-precert-signing-ca-sunset",
        "date": "2026-03-15",
        "title": "Precertificate Signing CAs PROHIBITED",
        "description": "SC-092: Dedicated Precertificate Signing CAs (Section 7.1.2.4 profile) can no longer issue certificates or precertificates. All precertificates must be signed by the same CA that signs final certificates. Simplifies CT ecosystem.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/09/02/ballot-sc-092-sunset-use-of-precertificate-signing-cas/",
        "category": "certificates",
        "isMajor": False,
        "impact": "Affects approximately 2 CAs with this configuration",
    },
    {
        "id": "mozilla-dual-purpose-plan",
        "date": "2026-04-15",
        "title": "Mozilla Dual-Purpose Root Transition Plans Due",
        "description": "MRSP v3.0: CA operators with existing dual-purpose roots (websites + email trust bits) must submit transition plan to Mozilla. Full migration to dedicated hierarchies required by December 31, 2028.",
        "source": "mozilla",
        "source_url": "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "category": "root-store",
        "isMajor": False,
        "impact": "Affects CAs with dual-purpose roots in Mozilla's root store"
    },
    {
        "id": "sectigo-clientauth-removal",
        "date": "2027-02-10",
        "title": "Sectigo Complete ClientAuth Removal",
        "description": "Sectigo no longer includes Client Authentication EKU in any SSL/TLS certificates. Hard deadline, no exceptions. Existing certificates remain valid until expiration or revocation.",
        "source": "chrome",
        "source_url": "https://www.sectigo.com/resource-library/deprecation-of-client-authentication-eku-from-sectigo-ssl-tls-certificates",
        "category": "eku",
        "isMajor": True,
        "consequences": {
            "enforcement": "Sectigo stops including the clientAuth EKU in all public TLS certificates \u2014 hard deadline, no exceptions. Existing certificates are unaffected until renewal.",
            "scenario": "An mTLS integration quietly breaks at renewal: the new cert passes every monitoring check (it is valid for serverAuth) but the partner's gateway rejects it on the EKU check. The failure looks like the partner's problem until someone diffs the old and new certs.",
        },
    },
    {
        "id": "digicert-clientauth-removal",
        "date": "2027-03-01",
        "title": "DigiCert Complete ClientAuth EKU Removal",
        "description": "DigiCert permanently removes Client Authentication EKU option from CertCentral for all public TLS certificates (DV, OV, EV, QWAC). Manual opt-in was available since the Oct 2025 default removal. Customers needing clientAuth should migrate to DigiCert X9 PKI or private CA.",
        "source": "chrome",
        "source_url": "https://knowledge.digicert.com/alerts/sunsetting-client-authentication-eku-from-digicert-public-tls-certificates",
        "category": "eku",
        "isMajor": True,
        "impact": "Organizations using DigiCert public TLS certs for mTLS must migrate before this date",
        "note": "DigiCert's timeline is more gradual than Sectigo (May 2026). Default removed Oct 1, 2025; manual opt-in available until March 1, 2027. Source: https://knowledge.digicert.com/alerts/sunsetting-client-authentication-eku-from-digicert-public-tls-certificates",
        "consequences": {
            "enforcement": "DigiCert permanently removes the clientAuth EKU option from CertCentral for all public TLS certificates (DV/OV/EV/QWAC).",
            "scenario": "Same failure mode as the Sectigo removal \u2014 mTLS breaks at renewal, not on the deadline date. Anything using a DigiCert public cert as a client identity needs to be on private PKI before its first post-March renewal.",
        },
    },
    {
        "id": "chrome-clientauth-leaf-sunset",
        "date": "2027-03-15",
        "title": "Chrome ClientAuth EKU Leaf Certificate Sunset",
        "description": "Chrome Root Program v1.8: Chrome will no longer trust newly issued leaf certificates containing both serverAuth and clientAuth EKUs. Subordinate CAs already restricted since June 2025. Deadline extended 9 months from the original June 2026 date to allow more time for enterprise migration.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "root-store",
        "isMajor": True,
        "impact": "mTLS with public certificates breaks - migrate to private CA",
        "note": "Date moved from 2026-06-15 to 2027-03-15 in late January/early February 2026. Chrome cited market feedback and the volume of concurrent 2026 PKI changes as reasons for the extension.",
        "consequences": {
            "enforcement": "Chrome will not trust newly issued leaf certificates carrying both serverAuth and clientAuth EKUs. Existing certificates are trusted until expiry.",
            "scenario": "This closes the last exit: even if a CA would still issue a dual-EKU cert, renewing one after this date means browsers reject your public-facing site. Dual-use certs need to be split \u2014 public cert for the site, private CA for the client identity \u2014 before their last pre-deadline renewal.",
        },
    },
    {
        "id": "chrome-ct-prelogging-required",
        "date": "2026-06-15",
        "title": "Chrome CT Pre-Logging Required",
        "description": "Chrome Root Program v1.8: CAs must ensure all TLS precertificates are logged to at least one CT log recognized by Chrome before issuing the corresponding final certificate.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "certificate-transparency",
        "isMajor": False,
        "impact": "CAs must adjust issuance pipeline to log before issuing"
    },
    {
        "id": "chrome-root-consolidation-plans",
        "date": "2026-06-15",
        "title": "Chrome Root Store Consolidation Plans Due",
        "description": "Chrome Root Program v1.8: CA Owners with more than two self-signed root CA certificates in the Chrome Root Store must submit a written consolidation plan identifying the two roots that will remain.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "root-store",
        "isMajor": False,
        "impact": "Affects CAs with >2 roots in Chrome Root Store"
    },
    {
        "id": "microsoft-secure-boot-expiry",
        "date": "2026-06-01",
        "status": "ongoing",
        "title": "Microsoft Secure Boot 2011 certificates begin expiring",
        "description": "Rolling expiration of the 2011-era Microsoft Secure Boot certificates begins June 2026 — not a single hard cutoff. Devices without 2023 certificates continue booting but lose access to new Secure Boot updates, boot manager updates, and revocation list updates. Affects BitLocker hardening and third-party bootloaders. Replacement certs: Microsoft UEFI CA 2023, Windows UEFI CA 2023, Microsoft Corporation KEK 2K CA 2023.",
        "source": "microsoft",
        "source_url": "https://support.microsoft.com/en-us/topic/windows-secure-boot-certificate-expiration-and-ca-updates-7ff40d33-95dc-4c3c-8725-a9b95457578e",
        "category": "platform",
        "isMajor": True,
        "is_estimated": True,
        "impact": "Any Windows device that misses the 2023 cert rollout loses the ability to receive Secure Boot security updates. Air-gapped and Windows 10 fleets are highest risk.",
        "framework_id": None,  # Secure Boot isn't a CA/B Forum thing; suppress the source=microsoft → cabforum default
        "framework_name": "Microsoft Root Program",
        "jurisdiction": "global",
        "consequences": {
            "enforcement": "Rolling expiry of the 2011-era Secure Boot certificates begins June 2026. Microsoft states devices without the 2023 certificates continue to start and operate normally and keep receiving standard Windows updates, but can no longer receive updates to Windows Boot Manager, the Secure Boot databases, revocation lists, or mitigations for newly discovered boot-level vulnerabilities.",
            "scenario": "Nothing breaks on the date, which is exactly the risk — the fleet keeps booting and the gap never shows up on a dashboard. Months later a boot-level vulnerability gets a revocation-list entry these devices cannot receive, and BitLocker hardening and third-party bootloader scenarios start behaving differently across the estate. Air-gapped systems and Windows 10 machines are the ones that quietly miss the 2023 certificate rollout.",
        },
    },
    {
        "id": "sc097-sha1-ca-crl-sunset",
        "date": "2026-09-15",
        "title": "SHA-1 Signatures Sunset (CA Certificates & CRLs)",
        "description": "SC-097: All CA certificates (roots and intermediates) signed with SHA-1 must be revoked. All CRL signing must migrate to SHA-256 or stronger. End-entity TLS certificates have required SHA-256+ since 2016 - this addresses remaining SHA-1 usage in CA infrastructure.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/01/24/ballot-sc097-sunset-all-remaining-use-of-sha-1-signatures-in-certificates-and-crls/",
        "category": "algorithm-deprecation",
        "isMajor": True,
        "impact": "CAs must revoke SHA-1 intermediates; organizations may need to update chain files",
        "ballotStatus": "passed",
        "ballotDetails": {
            "phaseLabel": "Effective",
            "votingStart": "2026-01-16T18:30:00Z",
            "votingEnd": "2026-01-23T18:30:00Z",
            "passedDate": "2026-01-24",
            "votingResults": {
                "certificateIssuers": {"yes": 26, "no": 0, "abstain": 0},
                "certificateConsumers": {"yes": 4, "no": 0, "abstain": 0},
                "quorumMet": True,
                "unanimous": True
            },
            "requirements": [
                "No new CRLs signed with SHA-1",
                "Revoke all unexpired Subordinate CA certificates with SHA-1 signatures"
            ],
            "exception": "SHA-1 still permitted for issuerKeyHash/issuerNameHash generation per RFC 5019",
            "proposers": ["Ryan Dickson (Google)", "Chris Clements (Google)"],
            "endorsers": ["Clint Wilson (Apple)", "Dimitris Zacharopoulos (HARICA)"],
            "sourceUrl": "https://github.com/cabforum/servercert/pull/645"
        },
        "consequences": {
            "enforcement": "All SHA-1-signed CA certificates (roots and intermediates) must be revoked and CRL signing must move to SHA-256 or stronger; CAs that miss it face misissuance findings and root-program scrutiny.",
            "scenario": "A legacy appliance shipping a stale CA bundle keeps presenting a chain through a now-revoked SHA-1 intermediate \u2014 TLS handshakes start failing on clients that check revocation, and the fix is a chain-file update nobody has owned in years.",
        },
    },
    {
        "id": "sc101v2-adn-derivation-mandatory",
        "date": "2026-11-15",
        "title": "Authorization Domain Name Derivation Rules Become Mandatory",
        "description": "SC-101v2 ('Clarify Authorization Domain Names'), adopted 2026-07-02 and published as TLS BR v2.2.9 effective 2026-08-06, replaces the descriptive Authorization Domain Name (ADN) definition with an explicit derivation algorithm in Section 3.2.2.4: the ADN must be selected AFTER the validation method is chosen; only methods validating at the name itself (not an underscore-prefixed subdomain) may use the 'cname' step; only methods that permit issuance for FQDNs ending in all the labels of the validated FQDN may use the 'prune' step; and CNAME-following must precede label-pruning where both occur. Per the transition provision, a CA may comply with Section 3.2.2.4 of BR v2.2.7 until 2026-11-15; from that date only Section 3.2.2.4 of the current Requirements applies. Also collapses wildcard and same-suffix method suitability into a table and simplifies the Base Domain Name and Domain Contact definitions.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/07/01/ballot-sc0101v2-clarify-authorization-domain-names/",
        "category": "validation",
        "isMajor": True,
        "impact": "CAs must implement the explicit ADN derivation algorithm; organizations that delegate domain control validation via CNAME should confirm their delegation pattern still validates under the method-specific cname/prune restrictions.",
        "ballotStatus": "passed",
        "ballotDetails": {
            "phaseLabel": "Effective",
            "passedDate": "2026-07-02",
            "votingResults": {
                "certificateIssuers": {"yes": 27, "no": 0, "abstain": 0},
                "certificateConsumers": {"yes": 4, "no": 0, "abstain": 0},
                "quorumMet": True,
                "unanimous": True,
            },
            "iprReviewPeriod": {
                "start": "2026-07-07T08:00:00Z",
                "end": "2026-08-06T08:00:00Z",
                "exclusionNoticeFiled": False,
            },
            "publishedIn": "TLS BR v2.2.9 (effective 2026-08-06)",
            "sourceUrl": "https://github.com/cabforum/servercert/blob/main/docs/BR.md",
        },
        "consequences": {
            "enforcement": "From 2026-11-15 the BR v2.2.7 fallback for Section 3.2.2.4 is withdrawn and CAs SHALL derive Authorization Domain Names using only the algorithm in the current Requirements. A CA that keeps issuing on an ADN derived the old way is misissuing, which is a revocation-and-incident-report matter under Section 4.9.1.1 and draws root-program scrutiny.",
            "scenario": "The change closes a real hole: under the old wording a CDN or other CNAME target could, on one reading, be validated for arbitrary subdomains of any name pointed at it. Most of the work here is CA-side, so for many teams there is nothing to do. The exception is worth checking now rather than in November — if you delegate domain control validation by CNAME (the common `_acme-challenge` delegation, or pointing a hostname at a vendor who validates on your behalf), the cname and prune steps are no longer available to every method. A delegation that renews cleanly today can start failing validation on 2026-11-15 with no change on your side, and the certificates it covers are typically the externally-facing ones you least want to discover this on.",
        },
    },
    {
        "id": "chrome-subca-automation-required",
        "date": "2027-03-15",
        "title": "Chrome Subordinate CA Automation Required",
        "description": "Chrome Root Program v1.8: All unexpired and unrevoked subordinate CA certificates signed by a root in the Chrome Root Store must be integrated with certificate lifecycle management automation solutions.",
        "source": "chrome",
        "source_url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "category": "automation",
        "isMajor": True,
        "impact": "All publicly-trusted subordinate CAs must support automation",
        "consequences": {
            "enforcement": "Every unexpired subordinate CA under a Chrome Root Store root must be integrated with certificate lifecycle automation; non-compliant hierarchies risk phase-out from the root store.",
            "scenario": "No direct action for most organizations \u2014 but a CA that cannot comply puts its whole hierarchy at phase-out risk, which becomes your problem if it is the hierarchy your certs chain to. A reasonable vendor-review question for your CA this year.",
        },
    },
    {
        "id": "validity-100-days",
        "date": "2027-03-15",
        "title": "MAX VALIDITY: 100 DAYS",
        "description": "Maximum certificate validity reduced to 100 days. Domain validation reuse reduced to 100 days.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/04/11/ballot-sc081v3-introduce-schedule-of-reducing-validity-and-data-reuse-periods/",
        "category": "certificates",
        "isMajor": True,
        "consequences": {
            "enforcement": "CAs cannot issue public TLS certificates valid longer than 100 days, and domain-validation reuse drops to 100 days (SC-081v3 schedule; drops again to 47 days in 2029).",
            "scenario": "Every manual renewal process goes from an annual calendar reminder to roughly four touches per certificate per year. A team managing 200 certs by spreadsheet goes from ~200 renewal events to ~800 \u2014 the missed-renewal outage stops being a question of if. This is the deadline that makes ACME automation a prerequisite rather than a nice-to-have.",
        },
    },
    {
        "id": "sc090-phone-validation-sunset",
        "date": "2027-03-15",
        "title": "Phone-Based Validation Methods SUNSET",
        "description": "SC-090: All phone-based DCV methods prohibited - 3.2.2.4.16 (DNS TXT Phone), 3.2.2.4.17 (DNS CAA Phone), 3.2.2.5.2 (Email/Fax/SMS/Mail to IP Contact), 3.2.2.5.5 (Phone to IP Contact). Transition to DNS/HTTP challenges required.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/20/ballot-sc-090-gradually-sunset-all-remaining-email-based-phone-based-and-crossover-validation-methods-from-sections-3.2.2.4-and-3.2.2.5/",
        "category": "validation",
        "isMajor": True,
        "impact": "Organizations using phone-based validation must migrate to automated methods",
        "consequences": {
            "enforcement": "All phone-, fax-, SMS-, and mail-based domain-control validation methods are prohibited (SC-090); CAs must use DNS/HTTP-based challenges only.",
            "scenario": "Organizations that lean on phone validation for oddball domains \u2014 acquired brands, domains whose DNS is controlled by another business unit \u2014 find they cannot renew at all until they get DNS or HTTP access sorted. The blocker is not technical, it is organizational, which is why it is worth starting a year early.",
        },
    },
    {
        "id": "sc091-reverse-lookup-sunset",
        "date": "2027-03-15",
        "title": "Reverse Address Lookup Validation SUNSET",
        "description": "SC-091: Method 3.2.2.5.3 ('Reverse Address Lookup') prohibited for IP address validation. Use new DNS TXT method (3.2.2.5.8) introduced Dec 2025, or other approved methods.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/12/ballot-sc-091-sunset-3.2.2.5.3-reverse-address-lookup-validation-proposal-of-new-dns-based-validation-using-persistent-dcv-txt-record-for-ip-addresses/",
        "category": "validation",
        "isMajor": False,
        "impact": "Affects IP address certificate validation workflows",
    },
    {
        "id": "sc090-email-validation-sunset",
        "date": "2028-03-15",
        "title": "Email-Based Validation Methods SUNSET",
        "description": "SC-090: All email-based DCV methods prohibited - 3.2.2.4.4 (Constructed Email), 3.2.2.4.13 (Email to DNS CAA), 3.2.2.4.14 (Email to DNS TXT). Only DNS and HTTP challenge methods remain. Full ACME/DNS automation required.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/20/ballot-sc-090-gradually-sunset-all-remaining-email-based-phone-based-and-crossover-validation-methods-from-sections-3.2.2.4-and-3.2.2.5/",
        "category": "validation",
        "isMajor": True,
        "impact": "Major migration required - email-based domain validation ends",
    },
    {
        "id": "legacy-dcv-complete-sunset",
        "date": "2028-03-15",
        "title": "ALL Legacy DCV Methods PROHIBITED",
        "description": "Complete sunset of all 11 legacy domain validation methods per SC-080, SC-090, SC-091. Email-based (5 methods), phone-based (4 methods), and reverse lookup (2 methods) all prohibited. Only DNS/HTTP challenge-response methods allowed.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/11/20/ballot-sc-090-gradually-sunset-all-remaining-email-based-phone-based-and-crossover-validation-methods-from-sections-3.2.2.4-and-3.2.2.5/",
        "category": "validation",
        "isMajor": True,
        "impact": "Full ACME automation required",
    },
    {
        "id": "mozilla-dual-purpose-migration-complete",
        "date": "2028-12-31",
        "title": "Mozilla Dual-Purpose Root Migration Complete",
        "description": "MRSP v3.0: All CA operators must complete migration from dual-purpose roots to dedicated TLS-only or S/MIME-only hierarchies. Roots must have one trust bit removed.",
        "source": "mozilla",
        "source_url": "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "category": "root-store",
        "isMajor": True,
        "impact": "All Mozilla-trusted roots must be single-purpose"
    },
    {
        "id": "validity-47-days",
        "date": "2029-03-15",
        "title": "MAX VALIDITY: 47 DAYS",
        "description": "Maximum certificate validity reduced to 47 days. Domain validation reuse reduced to 10 DAYS.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2025/04/11/ballot-sc081v3-introduce-schedule-of-reducing-validity-and-data-reuse-periods/",
        "category": "certificates",
        "isMajor": True,
        "impact": "Full automation REQUIRED",
    },
    {
        "id": "mozilla-mass-revocation-plan",
        "date": "2025-09-01",
        "title": "Mozilla Mass Revocation Plan Deadline",
        "description": "CAs must have documented mass revocation plans and procedures in place.",
        "source": "mozilla",
        "source_url": "https://wiki.mozilla.org/CA/Mass_Revocation_Events",
        "category": "revocation",
        "isMajor": True,
        "impact": "CAs without documented procedures may face removal from Mozilla root store.",
        "consequences": {
            "enforcement": "MRSP section 6.1.3: every CA operator capable of issuing TLS server certificates must have a comprehensive Mass Revocation Plan in place no later than 2025-09-01, covering activation criteria, current subscriber contact data, automation, timelines, annual testing, and an independent third-party assessment.",
            "scenario": "No action unless you operate a CA — but this is the plan that gets run against you. In a mass revocation your certificates are replaced on the CA's timetable (24 hours or 5 days under the Baseline Requirements), not yours. The enterprise-side questions are whether your CA holds current contact details for the team that would have to act, and whether you could actually reissue everything inside that window. Both are worth testing before a real event.",
        },
    },
    {
        "id": "mozilla-mass-revocation-assessment",
        "date": "2025-06-01",
        "title": "Mozilla Mass Revocation Assessment",
        "description": "Mozilla will assess CA readiness for mass revocation scenarios.",
        "source": "mozilla",
        "source_url": "https://wiki.mozilla.org/CA/Mass_Revocation_Events",
        "category": "revocation",
        "isMajor": False,
        "impact": "CAs must demonstrate capability to revoke large certificate volumes."
    },
    {
        "id": "mozilla-automation-disclosure",
        "date": "2025-04-15",
        "title": "Mozilla Automation Disclosure Requirement",
        "description": "CAs must disclose automation capabilities and ACME support status.",
        "source": "mozilla",
        "source_url": "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "category": "validation",
        "isMajor": False,
        "impact": "Transparency requirement for certificate automation readiness."
    },
    {
        "id": "chrome-entrust-distrust",
        "date": "2024-11-11",
        "status": "ongoing",
        "title": "Chrome Entrust Distrust",
        "description": "Chrome distrusts TLS certificates issued by Entrust after this date.",
        "source": "chrome",
        "source_url": "https://security.googleblog.com/2024/06/sustaining-digital-certificate-security.html",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will show security warnings in Chrome.",
        "consequences": {
            "enforcement": "Chrome blocks TLS server authentication certificates chaining to the listed Entrust and AffirmTrust roots whose earliest Signed Certificate Timestamp is dated after 2024-11-11 23:59:59 UTC; certificates with an earlier SCT stay trusted until they expire. Chrome's guidance is to move to a different publicly-trusted CA.",
            "scenario": "Anything still issued from an Entrust public root fails in Chrome the moment it is renewed. The trap is the long-lived certificate issued just before the cutoff: it works today and dies on renewal day, so the outage lands whenever the automation or the calendar reminder fires — not on a date anyone planned around.",
        },
    },
    {
        "id": "apple-entrust-distrust",
        "date": "2024-11-15",
        "status": "ongoing",
        "title": "Apple Entrust Distrust",
        "description": "Apple distrusts TLS certificates issued by Entrust after this date.",
        "source": "apple",
        "source_url": "https://support.apple.com/en-us/121668",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will not be trusted on macOS/iOS.",
        "consequences": {
            "enforcement": "Apple blocked the Entrust and AffirmTrust roots — including Entrust Root Certification Authority, G2, G4, EC1 and Entrust.net Certification Authority (2048) — on Apple platforms effective 2024-11-15, across TLS, S/MIME and timestamping uses.",
            "scenario": "Same migration as the Chrome distrust, wider blast radius. It is not just Safari: every app on macOS and iOS that uses the system trust store is affected, including background API clients that show no user-facing warning and simply start failing TLS. If a mobile app or any Apple-platform integration still terminates on an Entrust-issued certificate, it needs a different CA.",
        },
    },
    {
        "id": "mozilla-entrust-distrust",
        "date": "2024-11-30",
        "status": "ongoing",
        "title": "Mozilla Entrust Distrust",
        "description": "Firefox distrusts TLS certificates issued by Entrust after this date.",
        "source": "mozilla",
        "source_url": "https://groups.google.com/a/mozilla.org/g/dev-security-policy/c/jCvkhBjg9Yw",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will show security warnings in Firefox.",
        "consequences": {
            "enforcement": "Mozilla set TLS distrust-after dates on the Entrust roots in its root store: certificates issued after 2024-11-30 are not trusted by Firefox. Mozilla's stated reason was that Entrust's remediation plan was not sufficient to restore confidence in its operation.",
            "scenario": "Firefox is the browser nobody tests, so this one surfaces as a single support ticket rather than an alert. The rule is the same as for Chrome and Apple: any remaining Entrust public TLS certificate has to be replaced from a different CA at its next renewal, not left to run to expiry.",
        },
    },
    {
        "id": "microsoft-entrust-distrust",
        "date": "2025-04-16",
        "status": "ongoing",
        "title": "Microsoft Entrust Distrust",
        "description": "Microsoft distrusts TLS certificates issued by Entrust after this date.",
        "source": "microsoft",
        "source_url": "https://learn.microsoft.com/en-us/security/trusted-root/2025/february-2025",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will not be trusted in Windows/Edge.",
        "consequences": {
            "enforcement": "Microsoft's February 2025 Trusted Root Program release set a NotBefore date of 2025-04-16 on the Entrust and AffirmTrust roots — only certificates issued after that date are distrusted on Windows, and earlier certificates continue to validate.",
            "scenario": "This is the distrust that hits server-to-server traffic rather than browsers. A .NET or PowerShell client on Windows calling an Entrust-issued endpoint renewed after April 2025 throws a chain-validation error while the same endpoint works fine from Linux, which sends the investigation down the wrong path. Worth checking for Entrust certificates on internal endpoints that Windows hosts call.",
        },
    },
    # 2026-07-21 blind-window review: entries below recovered from the ~9 months
    # the Microsoft source pointed at a dead learn.microsoft.com page.
    {
        "id": "microsoft-april-2026-ctl-notbefore",
        "date": "2026-05-19",
        "status": "ongoing",
        "title": "Microsoft NotBefore distrust: SwissSign Silver G2, SecureSign RootCA11/CA12, ANCERT (new issuance)",
        "description": "Existing certificates keep working — renewals break. Certificates issued after May 19, 2026 that chain to the roots below fail Windows chain validation (CTL NotBefore, published in the April 2026 CTL release of Apr 28, 2026), which is why this often gets misdiagnosed as a CA outage: the certificate being replaced worked fine. Fully distrusted for new issuance (all uses): ANCERT root4; ANCERT root5; Byte Computer BYTE Root Certification Authority 001; Cybertrust Japan SecureSign RootCA11; Cybertrust Japan CA12; SwissSign Silver CA - G2. S/MIME-only NotBefore (email certificates issued after the date): AC Camerfirma Global Chambersign Root 2016; AC Camerfirma Chambers of Commerce Root 2016; Firmaprofesional firma2048; CFCA EV root (China Financial Certification Authority); ComSign root1; Cybertrust Japan CA14; Cybertrust Japan CA15; Cybertrust Japan iTrust Root Certification Authority; GDCA TrustAUTH R5 ROOT; GoDaddy GD Class 2 root; GoDaddy gdroot-g2; Starfield SF Class 2 root; Starfield sfroot-g2; Halcom Root Certificate Authority; HARICA 2015 RSA; HARICA 2015 ECC; NAVER Global Root Certification Authority; OATI oati_ca1; TrustFactory Client Root Certificate Authority. Disabled outright: AC Camerfirma CommerceRoot; Nets DanID (TDC) OCES Root. Certificates issued before May 19, 2026 remain trusted, and timestamped signatures continue to validate.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/trusted-root/2026/april-2026.md",
        "category": "certificates",
        "isMajor": True,
        "impact": "Renewals and new certificates under the affected roots fail Windows/Edge chain validation even though existing certificates keep working.",
        "consequences": {
            "enforcement": "Windows CTL NotBefore mechanism: certificates issued after 2026-05-19 from the listed roots fail chain validation on Windows; pre-existing certificates and timestamped signatures continue to validate.",
            "scenario": "A team renews a TLS certificate under SecureSign RootCA11 or SwissSign Silver CA - G2 in mid-2026. The renewed certificate is silently untrusted on Windows clients while the certificate it replaced worked fine — the failure only appears at renewal, so it gets misdiagnosed as a CA outage instead of a root-store distrust.",
        },
    },
    {
        "id": "microsoft-trp-single-purpose-roots",
        "date": "2026-07-01",
        "status": "ongoing",
        "title": "Microsoft TRP: single-purpose roots + 10-year max validity for new root submissions",
        "description": "Trusted Root Program Requirements v1.2 (May 20, 2026), effective for root certificates submitted on or after July 1, 2026: TLS server authentication, S/MIME, and code signing must be separate dedicated trust anchors (only Client Authentication may be combined, plus Time Stamping on code-signing roots), and newly minted roots are capped at 10 years validity from submission. Multi-EKU roots submitted before January 1, 2027 remain trusted unless Microsoft directs otherwise. v1.2 also mandates public incident disclosure in Bugzilla per the CCADB incident-report format.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/Requirements.md",
        "category": "root-store",
        "isMajor": False,
        "impact": "No action unless you operate a CA. Roots submitted on or after 2026-07-01 must be single-purpose and capped at 10 years; multi-EKU roots submitted before 2027-01-01 stay trusted, so CAs with multi-purpose hierarchies have until then to split them.",
    },
    {
        "id": "microsoft-august-2026-root-disable",
        "date": "2026-08-25",
        "title": "Microsoft disables GeoTrust Universal CA and the already-expired Baltimore CyberTrust Root; removes Visa Information Delivery Root CA",
        "description": "The August 2026 Trusted Root Program release (Tuesday, August 25, 2026) DISABLES two DigiCert roots outright — GeoTrust Universal CA (SHA-1 E621F3354379059A4B68309D8A2F74221587EC79) and Baltimore CyberTrust Root (D4DE20D05E66FC53FE1A50882C78DB2852CAE474) — and REMOVES Visa Information Delivery Root CA (SHA-256 C57A3ACBE8C06BA1988A83485BF326F2448775379849DE01CA43571AF357E74B). This is a different and harsher mechanism than the NotBefore distrust in the same release: a NotBefore only breaks certificates issued after its date, whereas disabling a root breaks every certificate under it regardless of issuance date. THE TWO DISABLED ROOTS DO NOT CARRY THE SAME CONSEQUENCE, and the naming order above is deliberate. GeoTrust Universal CA is valid 2004-03-04 to 2029-03-04 — it is live, so certificates chaining to it validate today and stop validating on Windows at the release whatever their issuance date. That is the real breakage here. Baltimore CyberTrust Root EXPIRED on 2025-05-12 at 23:59:00 GMT, fifteen months before this release: anything still chaining to it stopped validating in May 2025, so disabling it now is bookkeeping on a dead root and the operational impact is effectively nil. The residual Baltimore work is hygiene, not outage prevention — stale bundled trust stores and vendor/appliance images that still ship the root, monitoring or pinning that references its thumbprint, and Java cacerts copied off old hosts. NOTE on the date: the notice states one NotBefore date (2026-09-15) and lists Disable and Remove as separate actions of the release, without restating a date for them, so the release date is used here.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/trusted-root/2026/august-2026.md",
        "category": "certificates",
        "isMajor": True,
        "impact": "GeoTrust Universal CA is the one that can break working certificates: valid to 2029-03-04, and every certificate chaining to it stops validating on Windows — not just new issuance. Inventory GeoTrust chains before the release, not after. Baltimore CyberTrust Root expired 2025-05-12, so its disable is cleanup rather than an outage.",
        "consequences": {
            "enforcement": "Disabling removes the root's trust in the Windows CTL entirely, so all certificates chaining to it fail validation regardless of when they were issued. This is unlike the NotBefore sets in the same release, where certificates issued before the NotBefore date continue to validate. The mechanism is identical for both roots; the consequence is not, because an expired root has already been failing every chain it anchors. GeoTrust Universal CA is valid to 2029-03-04 and loses trust on the release date; Baltimore CyberTrust Root expired 2025-05-12 and lost it fifteen months earlier by expiry.",
            "scenario": "The failure lands where nobody is looking, and it is GeoTrust. A Windows host that has never been touched by the certificate team starts failing TLS to an internal appliance or a partner endpoint whose chain still terminates at GeoTrust Universal CA — and because the certificate itself did not change and did not expire, the first hypothesis is a network or firewall problem. Search the estate for chains ending at GeoTrust Universal CA before 2026-08-25 rather than diagnosing it afterwards from an outage. Do not spend the same effort on Baltimore CyberTrust Root: anything chaining to it broke in May 2025 when the root expired, so a Baltimore hit found now is an already-broken path or a stale trust-store copy, not a pending one — worth cleaning up, not worth a change freeze.",
        },
    },
    {
        "id": "microsoft-august-2026-ctl-notbefore",
        "date": "2026-09-15",
        "title": "Microsoft NotBefore distrust: Entrust/AffirmTrust 2022 roots, SecureTrust/Trustwave, Chunghwa Telecom (new issuance)",
        "description": "Existing certificates keep working — renewals break. Certificates issued after September 15, 2026 that chain to the roots below fail Windows chain validation (CTL NotBefore, published in the August 2026 release of Aug 25, 2026). Fully NotBefore'd for all uses (19 roots): Carillon PKI Services G2 Root CA 1; certSIGN_root; DigiCert Hotspot 2.0 Trust Root CA - 03; Entrust AffirmTrust Commercial, AffirmTrust Networking, AffirmTrust Premium, AffirmTrust Premium ECC, AffirmTrust 4K TLS Root CA - 2022, Entrust 4K TLS Root CA - 2022, Entrust 4K EV TLS Root CA - 2022, Entrust P384 TLS Root CA - 2022, Entrust P384 EV TLS Root CA - 2022; SecureTrust XRamp Global Certification Authority, SecureTrust CA, Secure Global CA, Trustwave Global Certification Authority, Trustwave Global ECC P256, Trustwave Global ECC P384; Visa Public RSA Root CA. Per-EKU NotBefores, which is where this gets misread — the same root can be distrusted for one use and fine for another: code signing (Camerfirma Chambers of Commerce Root - 2008; SECOM_SCRoot2; Security Communication RootCA3); S/MIME (Camerfirma Chambers of Commerce Root - 2008; Chunghwa Telecom ePKI Root CA - G2, ePKI Root CA - G4 and CHT_eCA; GovSaudiArabia NCDC; Swedish Government Root Authority v3; both Notarius Root Certificate Authority roots; OISTE WISeKey Global Root GC CA; Thailand National Root CA - G1); time stamping (Chunghwa Telecom ePKI G2, ePKI G4, CHT_eCA; OISTE WISeKey Global Root GC CA); server authentication (Chunghwa Telecom ePKI G2 and CHT_eCA; Swedish Government Root Authority v3); client authentication (Chunghwa Telecom ePKI G2 and CHT_eCA); document signing (OISTE WISeKey Global Root GC CA). The same release also adds 8 ML-DSA PQC pilot roots (ComSign, DigiCert, HARICA, IdenTrust, Sectigo, UniTrust/Shanghai, SSL.com, Visa) and folds the Certificate Transparency Log Monitor (CTLM) policy into the monthly Windows CTL, with opt-in SCT validation currently event-logging only. Certificates issued before September 15, 2026 remain trusted, and timestamped signatures continue to validate.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/trusted-root/2026/august-2026.md",
        "category": "certificates",
        "isMajor": True,
        "impact": "Renewals and new certificates under the affected roots fail Windows/Edge chain validation even though existing certificates keep working. Check per-EKU: several roots are distrusted for some uses and not others.",
        "consequences": {
            "enforcement": "Windows CTL NotBefore mechanism: certificates issued after 2026-09-15 from the listed roots fail chain validation on Windows for the affected uses; pre-existing certificates and timestamped signatures continue to validate.",
            "scenario": "The per-EKU split is the trap. Chunghwa Telecom's CHT_eCA and ePKI Root CA - G2 are NotBefore'd for S/MIME, time stamping, server authentication AND client authentication — effectively everything — while ePKI Root CA - G4 is hit only for S/MIME and time stamping. So a team that checks one root, concludes 'this CA is distrusted', and migrates everything gets a different answer than a team that checks the EKU it actually uses. Read the per-use lists, not the root name. Separately, this catches the Entrust 2022-vintage roots, which are a different set from the 2025-04-16 Entrust distrust already tracked — an organization that moved onto a 2022 Entrust root after that event needs to check whether it is now in this list.",
        },
    },
    # =============================================================================
    # POST-QUANTUM CRYPTOGRAPHY (PQC) / CNSA 2.0
    # =============================================================================
    {
        "id": "nist-pqc-standards",
        "date": "2024-08-13",
        "title": "NIST PQC Standards Published",
        "description": "NIST released FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), and FIPS 205 (SLH-DSA). Organizations should begin planning migration to quantum-resistant cryptography.",
        "source": "nist",
        "source_url": "https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards",
        "category": "pqc",
        "isMajor": True,
        "impact": "Start inventory and planning now",
        "consequences": {
            "enforcement": "FIPS 203 (ML-KEM), FIPS 204 (ML-DSA) and FIPS 205 (SLH-DSA) are final federal standards. Publication carries no compliance deadline of its own; NIST's guidance is to start integrating them immediately because full integration takes time. Binding dates come from separate mandates such as CNSA 2.0 and EO 14412 / OMB M-26-15.",
            "scenario": "Nothing fails on this date — the cost of ignoring it is paid later. Harvest-now-decrypt-later means traffic captured today becomes readable once a cryptographically relevant quantum computer exists, so long-lived confidential data protected only by RSA or ECDH key exchange is already exposed. The first practical step is a cryptographic inventory: where RSA and ECC are used, and which of those uses sit in hardware or vendor products you cannot change quickly.",
        },
    },
    {
        "id": "microsoft-pqc-tls-pilot",
        "date": "2026-06-08",
        "title": "Microsoft launches PQC TLS Pilot Program (ML-DSA-87 roots)",
        "description": "Trusted Root Program PQC TLS Pilot V1.0: approved participating CAs may operate one ML-DSA-87 pilot root (ML-DSA-44/65 permitted below the root) for TLS server/client authentication testing in closed environments. Pilot certificates are NOT publicly trusted, not CT-logged, and not in CCADB; validity caps are 1 year for roots and subordinates, 90 days for leaves.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/PQC%20Pilot%20Program.md",
        "category": "pqc",
        "isMajor": False,
        "impact": "Informational — Windows-side PQC TLS interoperability testing has begun; no production action for enterprises.",
    },
    {
        "id": "cnsa-2-software-signing",
        "date": "2025-12-31",
        "title": "CNSA 2.0: Software Signing Prefer PQC",
        "description": "NSA recommends software and firmware signing should support and prefer CNSA 2.0 (ML-DSA) algorithms by end of 2025.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "cnsa-2-web-prefer",
        "date": "2025-12-31",
        "title": "CNSA 2.0: Web Browsers/Servers Prefer PQC",
        "description": "NSA recommends web browsers, servers, and cloud services support and prefer CNSA 2.0 algorithms by end of 2025.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "cnsa-2-network-equipment",
        "date": "2026-12-31",
        "title": "CNSA 2.0: Network Equipment Prefer PQC",
        "description": "NSA recommends VPNs, routers, and traditional network equipment support and prefer CNSA 2.0 algorithms by end of 2026.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "cnsa-2-operating-systems",
        "date": "2027-12-31",
        "title": "CNSA 2.0: Operating Systems Prefer PQC",
        "description": "NSA recommends operating systems support and prefer CNSA 2.0 algorithms by end of 2027.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": False,
    },
    {
        "id": "cnsa-2-software-exclusive",
        "date": "2030-12-31",
        "title": "CNSA 2.0: Software Signing Exclusive PQC",
        "description": "NSA target for software and firmware signing to use CNSA 2.0 algorithms exclusively. Traditional RSA/ECDSA signatures no longer acceptable for NSS.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": True,
        "impact": "Code signing must use ML-DSA",
    },
    {
        "id": "cnsa-2-network-exclusive",
        "date": "2030-12-31",
        "title": "CNSA 2.0: Network Equipment Exclusive PQC",
        "description": "NSA target for VPNs, routers, and network equipment to use CNSA 2.0 algorithms exclusively. Traditional public key cryptography no longer acceptable.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": True,
        "impact": "Traditional RSA/ECC deprecated for key exchange",
    },
    {
        "id": "nist-weak-algo-deprecated",
        "date": "2030-12-31",
        "title": "NIST Deprecates Weak Algorithms",
        "description": "Per NIST SP 800-131A Rev 3, SHA-1, SHA-224, and AES-ECB mode are deprecated. Minimum security strength increases from 112 bits to 128 bits. DRAFT SOURCE: Rev 3 remains an initial public draft (published 2024-10-21, comment period closed 2024-12-04); no final has been issued as of 2026-08-05, and /pubs/sp/800/131/a/r3/final returns 404. Treat the date as NIST's stated direction, not a published federal requirement.",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/sp/800/131/a/r3/ipd",
        "category": "pqc",
        "isMajor": True,
        "is_estimated": True,
        "impact": "RSA-2048 no longer meets minimum requirements",
    },
    {
        "id": "cnsa-2-web-exclusive",
        "date": "2033-12-31",
        "title": "CNSA 2.0: Web/Cloud Exclusive PQC",
        "description": "NSA target for web browsers, servers, cloud services, and operating systems to use CNSA 2.0 algorithms exclusively.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": True,
        "impact": "Full PQC transition required for web PKI",
    },
    {
        "id": "nist-pqc-transition-complete",
        "date": "2035-12-31",
        "title": "NIST Removes Quantum-Vulnerable Algorithms",
        "description": "Under NIST IR 8547 timeline, quantum-vulnerable algorithms (RSA, ECDSA, DH, ECDH) will be removed from NIST standards. DRAFT SOURCE: IR 8547 remains an initial public draft (published 2024-11-12, comment period closed 2025-01-10); no final has been issued as of 2026-08-05, and /pubs/ir/8547/final returns 404. The report describes NIST's expected approach, so this is directional rather than binding.",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/ir/8547/ipd",
        "category": "pqc",
        "isMajor": True,
        "is_estimated": True,
        "impact": "RSA/ECC no longer in NIST standards",
    },
    {
        "id": "cabf-sc099-validation-logging",
        "date": "2026-07-15",
        "title": "CA/B Forum SC099 — Validation logging requirements effective",
        "description": "CA validation logging requirements per SC099 become operationally effective. CAs must log validation evidence for audit purposes.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/04/18/ballot-sc099-improve-recording-of-validation-method/",
        "category": "validation",
        "isMajor": False,
        "framework_id": "cabforum",
        "framework_name": "CA/Browser Forum",
        "jurisdiction": "global",
    },
    {
        "id": "fips-140-2-historical",
        "date": "2026-09-21",
        "title": "FIPS 140-2 validations move to Historical list (CMVP)",
        "description": "NIST CMVP moves all remaining active FIPS 140-2 certificates to Historical status. Modules keep running, but federal agencies should not include them in new procurements. Affects CMMC, DFARS, FedRAMP evidence chains. Full retirement: 2031-09-21 (five years after Historical).",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/projects/fips-140-3-transition-effort",
        "category": "algorithm-deprecation",
        "isMajor": True,
        "impact": "Federal procurement chains. CMMC Level 2 enforcement follows seven weeks later. Audit evidence weakens for any system running FIPS 140-2-only modules.",
        "framework_id": "nist",
        "framework_name": "NIST",
        "jurisdiction": "us",
        "consequences": {
            "enforcement": "NIST CMVP moves remaining FIPS 140-2 certificates to Historical status; federal agencies should not include Historical modules in new procurements. Modules keep running \u2014 nothing breaks at runtime.",
            "scenario": "A contract renewal or ATO package citing a 140-2-only module gets flagged by the assessor; CMMC Level 2 and FedRAMP evidence chains need re-papering against FIPS 140-3 validations, which have long queues.",
        },
    },
    {
        "id": "cnsa-2-nss-acquisition-begins",
        "date": "2027-01-01",
        "title": "CNSA 2.0 — NSS acquisition requirement begins",
        "description": "NSA Commercial National Security Algorithm Suite 2.0. National Security Systems must begin acquiring CNSA 2.0-compliant products where available. Full transition target: 2033.",
        "source": "nsa",
        "source_url": "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF",
        "category": "pqc",
        "isMajor": False,
        "framework_id": "nsa",
        "framework_name": "NSA CNSA 2.0",
        "jurisdiction": "us",
    },
    {
        "id": "cabf-sc098v2-caa-parameters",
        "date": "2027-03-15",
        "title": "CA/B Forum SC098v2 — CAA parameter processing (RFC 8657) required",
        "description": "Ballot SC098v2 ('Process RFC 8657 CAA Parameters', TLS BR 2.2.8) requires CAs to process the RFC 8657 CAA parameters accounturi and validationmethods rather than ignoring them, and defines required syntax for non-ACME validation methods. Passed 2026-05-11 (22-1); IPR review completed 2026-06-12. Requirements take effect March 15, 2027.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/05/13/ballot-sc098v2-process-rfc-8657-caa-parameters/",
        "category": "validation",
        "isMajor": True,
        "impact": "CAs must implement RFC 8657 CAA parameter processing (accounturi, validationmethods) prior to certificate issuance by this date.",
        "framework_id": "cabforum",
        "framework_name": "CA/Browser Forum",
        "jurisdiction": "global",
        "consequences": {
            "enforcement": "CAs must process RFC 8657 CAA parameters (accounturi, validationmethods) instead of ignoring them.",
            "scenario": "Cuts both ways. Configured well, your CAA records become a real control \u2014 issuance is pinned to your ACME account, so a stolen DNS foothold elsewhere cannot mint certs for your domains. Configured badly, a typo'd accounturi silently blocks your own renewals starting on the effective date. Audit CAA records before March 2027.",
        },
    },
    {
        "id": "chrome-root-consolidation-phaseout",
        "date": "2027-09-15",
        "title": "Chrome Root Store two-roots-per-CA-Owner cap takes effect",
        "description": "Chrome Root Program Policy v1.8 §1.2.1: consolidation plans (due 2026-06-15) from CA Owners with more than two self-signed roots must declare phase-out dates for excess roots falling before September 15, 2027 (00:00 UTC); effective that date the Chrome Root Store enforces a maximum of two self-signed root CA certificates per CA Owner. Roots in active phase-out are excluded from the cap and case-by-case extensions are possible. Phase-out is an SCTNotAfter constraint: certificates issued before the phase-out date remain trusted until expiry.",
        "source": "chrome",
        "source_url": "https://googlechrome.github.io/chromerootprogram/#121-maximum-number-of-cas-per-ca-owner",
        "category": "root-store",
        "isMajor": True,
        "impact": "CA Owners with more than two roots in the Chrome Root Store must have excess roots in phase-out; certificates issued from phased-out roots after their declared dates are not trusted by default.",
        "consequences": {
            "enforcement": "The Chrome Root Store enforces a maximum of two self-signed roots per CA Owner; certificates issued from a phased-out root after its declared date are not trusted by default (already-issued certs remain trusted until expiry).",
            "scenario": "If your CA is consolidating, renewals may start chaining to a different root than the one your infrastructure expects. Anything that pins roots \u2014 mobile apps, IoT fleets, agent software with bundled trust stores \u2014 breaks not on the deadline, but on the first renewal that lands on the new chain. Ask your CA which roots survive and check your pins against that list.",
        },
    },
    {
        "id": "nist-800-131a-112bit-disallowed",
        "date": "2030-12-31",
        "title": "NIST SP 800-131A Rev 3 — 112-bit security strength disallowed",
        "description": "End of approved use of 112-bit security strength for federal applications. Affects RSA-2048, ECC P-224, and 3-key 3DES. Drives minimum to 128-bit equivalent (RSA-3072 / ECC P-256 or stronger). DRAFT SOURCE: Rev 3 remains an initial public draft (published 2024-10-21, comment period closed 2024-12-04); no final has been issued as of 2026-08-05, and /pubs/sp/800/131/a/r3/final returns 404. Treat the date as NIST's stated direction, not a published federal requirement.",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/sp/800/131/a/r3/ipd",
        "category": "algorithm-deprecation",
        "isMajor": True,
        "is_estimated": True,
        "impact": "RSA-2048 — the de facto default for the past two decades — becomes unacceptable for federal use. Plan key-size migrations now; certificates valid past 2030 should already be RSA-3072+ or ECC P-256+.",
        "framework_id": "nist",
        "framework_name": "NIST SP 800-131A",
        "jurisdiction": "us",
    },
    {
        "id": "chrome-dedicated-tls-enforcement",
        "date": "2026-06-15",
        "title": "Chrome begins phasing out non-dedicated-TLS hierarchies",
        "description": "Chrome Root Program Policy v1.8 §1.3.2: beginning June 15, 2026, Chrome phases out PKI hierarchies found in violation of dedicated-TLS requirements — sub-CAs disclosed on/after this date must assert only id-kp-serverAuth. Violations detected after this date get a phase-out date set 90 calendar days following detection (Chrome retains case-by-case discretion). Distinct from the March 15, 2027 leaf-certificate dual-EKU sunset.",
        "source": "chrome",
        "source_url": "https://googlechrome.github.io/chromerootprogram/#132-promote-use-of-dedicated-tls-server-authentication-pki-hierarchies",
        "category": "root-store",
        "isMajor": False,
        "impact": "CA hierarchies mixing TLS with other use cases are now subject to rolling 90-day phase-outs on detection.",
        "status": "ongoing",
        "consequences": {
            "enforcement": "Hierarchies found violating the dedicated-TLS requirements receive a phase-out date 90 calendar days after detection; certificates issued after that date are not trusted by default in Chrome.",
            "scenario": "No action for most organizations unless you operate a CA. Indirect exposure: if your CA's hierarchy is phased out, your renewals may need to move to a different hierarchy — worth asking your CA whether their sub-CAs are serverAuth-only.",
        },
    },
    {
        "id": "omb-m-26-15-pqc-plan-due",
        "date": "2026-10-22",
        "title": "OMB M-26-15: agency PQC migration plans due to OMB and ONCD",
        "description": "OMB Memo M-26-15 ('Execution of the Migration to Post-Quantum Cryptography', June 24, 2026) requires executive agencies to submit PQC migration plans to OMB and ONCD no later than 120 days from issuance (~October 22, 2026). Plans must follow the memo's five-phase schedule (discovery 2026-27, pilots 2027-28, prioritized key-establishment migration 2028-30, signature migration 2031, full migration by 2035), align with NIST IR 8547, and cover inventory tooling, crypto-agility, third parties, and funding. Does not apply to national security systems.",
        "source": "nist",
        "source_url": "https://www.whitehouse.gov/wp-content/uploads/2026/06/M-26-15-Execution-of-the-Migration-to-Post-Quantum-Cryptography.pdf",
        "category": "pqc",
        "isMajor": True,
        "impact": "Federal agencies must deliver a compliant PQC migration plan; contractors and vendors should expect inventory and crypto-agility questions to flow down.",
        "consequences": {
            "enforcement": "M-26-15 directs agency heads to submit migration plans within 120 days; OMB and ONCD track submission and plan content against Appendix B requirements as part of FISMA oversight.",
            "scenario": "An agency without a certificate and cryptography inventory cannot write a credible plan — the 120-day clock effectively makes crypto discovery a Q3 2026 fire drill. Vendors to federal agencies should expect data calls about PQC readiness and TLS 1.3 support as agencies scramble to fill Appendix B sections.",
        },
    },
    {
        "id": "federal-tls13-support-required",
        "date": "2030-01-02",
        "title": "Federal agencies must support TLS 1.3 (M-26-15 / EO 14306)",
        "description": "OMB M-26-15 Appendix A §4 (restating EO 14306): agencies must support TLS 1.3 or a successor version no later than January 2, 2030. Agency PQC migration plans must include milestones toward this date.",
        "source": "nist",
        "source_url": "https://www.whitehouse.gov/wp-content/uploads/2026/06/M-26-15-Execution-of-the-Migration-to-Post-Quantum-Cryptography.pdf",
        "category": "pqc",
        "isMajor": False,
        "impact": "Federal systems and services sold to agencies must support TLS 1.3 — a prerequisite for PQC key establishment (ML-KEM runs over TLS 1.3).",
    },
    {
        "id": "omb-m-26-15-full-pqc-migration",
        "date": "2035-12-31",
        "title": "OMB M-26-15 Phase 5: full federal PQC migration target",
        "description": "Final phase of the M-26-15 five-phase schedule: agencies should complete migration of remaining systems to PQC by 2035, risk-based and contingent on commercial availability. A planning target ('should'), not a hard mandate; the binding earlier milestones are 2030 (key establishment) and 2031 (signatures) per EO 14412.",
        "source": "nist",
        "source_url": "https://www.whitehouse.gov/wp-content/uploads/2026/06/M-26-15-Execution-of-the-Migration-to-Post-Quantum-Cryptography.pdf",
        "category": "pqc",
        "isMajor": False,
        "impact": "End-state target for the federal PQC transition; aligns with the CNSA 2.0 exclusive-use horizon.",
        "is_estimated": True,
    },
    {
        "id": "cabf-csc32-reserved-policy-oid",
        "date": "2026-09-15",
        "title": "CA/B Forum CSC-32 — reserved policy OID mandatory in code signing certificates",
        "description": "Code Signing BRs v3.11.0 §7.1.6.4 (ballot CSC-32, passed 2026-05-11, IPR completed 2026-06-10 with no exclusions): effective September 15, 2026, every newly issued publicly trusted code signing subscriber certificate MUST contain exactly one of the reserved policy OIDs in certificatePolicies — 2.23.140.1.4.1 (non-EV code signing), 2.23.140.1.3 (EV code signing), 2.23.140.1.4.2 (timestamping). CA-defined policy OIDs may additionally be present.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/06/16/ballot-csc-32-make-a-reserved-policy-oid-mandatory/",
        "category": "certificates",
        "isMajor": False,
        "impact": "CAs must include the correct reserved policy OID at issuance; tooling that inspects certificatePolicies gains a reliable discriminator for code-signing certificate types.",
        "framework_id": "cabforum",
        "framework_name": "CA/Browser Forum",
        "jurisdiction": "global",
    },
    {
        "id": "smc017-smime-ca-rsa-4096",
        "date": "2026-09-15",
        "title": "S/MIME Root and Subordinate CA RSA keys must be 4096-bit",
        "description": "S/MIME BRs v1.0.15 (ballot SMC017v2, passed 2026-06-16; IPR Review Period 2026-06-30 20:00 UTC to 2026-07-30 20:00 UTC closed with no Exclusion Notice, v1.0.15 published 2026-07-30): the minimum RSA key size for Root and Subordinate CA certificates rises from 2048 to 4096 bits for keys CREATED after September 15, 2026. The trigger is the key creation date, not the certificate issuance date — key material generated on or before that date remains usable under the 2048-bit minimum. Subscriber certificates are unaffected and retain the 2048-bit minimum.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/06/16/ballot-smc-017v2/",
        "category": "algorithm-deprecation",
        "isMajor": True,
        "impact": "S/MIME CAs must generate Root and Subordinate CA keys at RSA-4096 or stronger after 2026-09-15; key ceremonies, HSM capacity and hierarchy planning need to account for the larger keys.",
        "is_estimated": False,
        "framework_id": "cabforum",
        "framework_name": "CA/Browser Forum",
        "jurisdiction": "global",
        "consequences": {
            "enforcement": "From 2026-09-15 the S/MIME BRs require Root and Subordinate CA certificates to use RSA keys of at least 4096 bits where the key was created after that date. Because the requirement keys off key creation, a CA that pre-generated 2048-bit CA key material before the date is not retroactively non-compliant, but any key generated afterwards must meet the new floor. Subscriber certificate keys keep the 2048-bit minimum.",
            "scenario": "No action unless you operate an S/MIME CA — but the key-creation trigger is worth reading carefully if you do, because it cuts both ways. A CA that ran a key ceremony in August 2026 and holds the keys in reserve can still issue from them; a CA that assumed the rule tracked issuance date and scheduled its ceremony for October at 2048 bits has a problem it will not see until an audit. For enterprises the practical effect arrives second-hand: S/MIME hierarchies get rebuilt on larger keys, so expect new intermediates and chain updates in mail clients and gateways over the following year.",
        },
    },
    {
        "id": "smc017-smime-subca-3072-issuance-sunset",
        "date": "2027-09-15",
        "title": "No S/MIME Subscriber issuance from Sub-CAs with RSA modulus under 3072",
        "description": "S/MIME BRs v1.0.15 (ballot SMC017v2): by September 15, 2027, CAs SHALL NOT issue Subscriber certificates from any Subordinate CA whose RSA key modulus is less than 3072 bits — sunsetting issuance from legacy 2048-bit Sub-CAs. This is a restriction on issuance from the Sub-CA, not on the Subscriber key size, and it is separate from the 2026-09-15 requirement that newly created Root/Sub-CA keys be 4096-bit.",
        "source": "cab-forum",
        "source_url": "https://cabforum.org/2026/06/16/ballot-smc-017v2/",
        "category": "algorithm-deprecation",
        "isMajor": True,
        "impact": "S/MIME CAs still operating 2048-bit Subordinate CAs must stand up replacement Sub-CAs at 3072 bits or higher and migrate issuance before 2027-09-15.",
        "is_estimated": False,
        "framework_id": "cabforum",
        "framework_name": "CA/Browser Forum",
        "jurisdiction": "global",
        "consequences": {
            "enforcement": "From 2027-09-15 a CA may not issue S/MIME Subscriber certificates from a Subordinate CA whose RSA modulus is under 3072 bits. Existing certificates already issued from those Sub-CAs are not revoked by the rule, but the issuing path closes — continued issuance requires a Sub-CA meeting the new floor.",
            "scenario": "The failure mode is a renewal that stops working rather than a certificate that breaks. An organization whose S/MIME certificates come from a long-lived 2048-bit intermediate finds that renewals after September 2027 arrive under a different intermediate, which matters wherever that chain was pinned or manually installed — mail gateways, signing appliances, archived trust bundles. Worth asking your S/MIME provider now which intermediate you are issuing from and what its replacement will be.",
        },
    },
    {
        "id": "digicert-g1-root-distrust",
        "date": "2026-04-15",
        "status": "ongoing",
        "title": "DigiCert G1 roots distrusted by Chrome and Mozilla",
        "description": "On April 15, 2026 Chrome and Mozilla removed DigiCert's first-generation roots — DigiCert Assured ID Root CA, DigiCert Global Root CA, DigiCert High Assurance EV Root CA — from their trust stores. Full removal: ALL TLS certificates chaining solely to G1 roots lost trust regardless of issuance date; reissuance under Global Root G2 (RSA) or G3 (ECC) required. Distinct from the Chrome-only Trusted Root G4 / Assured ID G2/G3 action (SCTNotAfter-style, certs issued on/after 2026-07-01); related to DigiCert's 2026-05-15 self-revocation of non-TLS G2/G3 ICAs and G5 cross-signed roots — all part of the dedicated-TLS-hierarchy alignment.",
        "source": "chrome",
        "source_url": "https://knowledge.digicert.com/alerts/digicert-root-strategy-aligning-with-industry-standards",
        "category": "root-store",
        "isMajor": True,
        "impact": "Any service still presenting a chain to a G1 root shows trust errors in Chrome and Firefox since April 15, 2026 — reissue under G2/G3.",
        "consequences": {
            "enforcement": "Chrome and Mozilla removed the three G1 roots outright: every TLS certificate chaining solely to them is untrusted regardless of issuance date — unlike SCTNotAfter phase-outs, existing certificates were not grandfathered.",
            "scenario": "The stragglers' failure mode: an internal service or appliance still serving the old G1 chain throws NET::ERR_CERT_AUTHORITY_INVALID for anyone on a current browser. If you find one, the cert itself is usually fine — re-download the G2/G3 chain from DigiCert and replace the bundle.",
        },
    },
    {
        "id": "luxembourg-nis2-in-force",
        "date": "2026-05-10",
        "title": "Luxembourg NIS2 transposition — entry into force",
        "description": "Loi du 5 mai 2026 concernant des mesures destinées à assurer un niveau élevé de cybersécurité. Published 5 May 2026, entered into force 10 May 2026. NIS2 obligations now apply to ~6,000–8,000 Luxembourg entities (essential and important). Self-registration via ILR portal.",
        "source": "nis2",
        "source_url": "https://legilux.public.lu/eli/etat/leg/loi/2026/05/05/a225/jo",
        "category": "platform",
        "isMajor": False,
        "framework_id": "nis2",
        "framework_name": "NIS2",
        "jurisdiction": "eu",
        "status": "ongoing",
    },
    # Apple Root Program Policy v2.0 (applied 2026-08-15). Source of all three:
    # https://github.com/apple/apple-root-program/blob/main/policy.md — the
    # document's own Change Log row for v2.0 dates every obligation below, and
    # the normative sections restate the dates except where noted per entry.
    # Split three ways because the dates differ, not the mechanisms.
    {
        "id": "apple-policy-v2-subca-eku",
        "date": "2026-08-01",
        "status": "ongoing",
        "title": "Apple Policy v2.0: Sub-CA EKU mandatory, Apple approval for Externally Operated Sub-CAs, RSA-4096/ECDSA-384 floor for new roots",
        "description": "Apple Root Program Policy v2.0 (header: \"Effective 2026-08-01\") starts three obligations on this date. (1) §1.7: a Subordinate CA Certificate signed on or after 2026-08-01 MUST contain an Extended Key Usage extension and MUST NOT assert anyExtendedKeyUsage (2.5.29.37.0). One signed on or after 2026-08-01 and before 2027-07-01 must additionally satisfy either (a) dedication to a single Trust Purpose per Appendix A, or (b) issuance under a multi-purpose root with no Appendix A Trust-Purpose EKU at all and its use case (e.g. Document Signing) covered by a CP/CPS inside the annual audit scope; Apple may require issuance from a (b) Sub-CA to stop at any time. For this section a renewal, re-key or cross-sign produces a NEW Subordinate CA Certificate whose signing date is the date of that issuance. (2) §1.4: CA Owners MUST receive Apple's approval prior to issuing each Subordinate CA Certificate (or cross-signed certificate) to an Externally Operated Subordinate CA. (3) §1.5: Root Inclusion Requests MUST only contain Root CA Certificates with a minimum key size of RSA 4096-bit or ECDSA 384-bit, and are accepted only for hierarchies dedicated to a single Trust Purpose with a single combined CP/CPS in Markdown. Per the §2 note, issuance-related effective dates are enforced from 00:00:00 UTC on the stated date.",
        "source": "apple",
        "source_url": "https://github.com/apple/apple-root-program/blob/main/policy.md",
        "category": "root-store",
        "isMajor": True,
        "impact": "Every Sub-CA certificate Apple sees from 2026-08-01 onward — including renewals, re-keys and cross-signs of existing Sub-CAs — must carry an EKU extension and must not assert anyExtendedKeyUsage.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "Apple Root Program Policy v2.0 §1.7 binds every Subordinate CA Certificate signed on or after 2026-08-01: EKU extension present, anyExtendedKeyUsage absent, and single-Trust-Purpose dedication unless the narrow (b) carve-out applies. Non-conformance is a policy violation handled through the program's incident process (§1.2.4, §3), and Apple reserves the right to require a CA Owner to discontinue issuance from a Sub-CA qualifying under (b). The §1.4 approval duty and the §1.5 key-size floor are dated by the v2.0 Change Log; their section text states the requirement without restating the date.",
            "scenario": "No action unless you operate a CA — but the date is not a one-off, which is why it stays in force rather than reading as completed. It re-applies every time a Sub-CA is renewed, re-keyed or cross-signed, because the policy treats each of those as a newly signed Subordinate CA Certificate with a fresh signing date. A multi-purpose Sub-CA that was compliant when issued in 2025 fails this rule the day it is re-keyed in 2027 unless it has been split by Trust Purpose first. For an enterprise buying from a public CA, the visible effect is downstream: your CA's hierarchy gets split per Trust Purpose, so the issuing CA behind your TLS certificates may change identity at renewal even though nothing about your certificate request did.",
        },
    },
    {
        "id": "apple-policy-v2-smime-rfc822name",
        "date": "2027-02-01",
        "title": "Apple: newly signed S/MIME subscriber certificates must carry an rfc822Name SAN",
        "description": "Apple Root Program Policy v2.0 §2.3: effective 2027-02-01, all newly signed Subscriber certificates that contain the id-kp-emailProtection EKU MUST include at least one rfc822Name in the subjectAltName extension. Applies to the Secure Email and Legacy S/MIME Trust Purposes. Per the §2 note the requirement is enforced for certificates signed on or after 2027-02-01 at 00:00:00 UTC.",
        "source": "apple",
        "source_url": "https://github.com/apple/apple-root-program/blob/main/policy.md",
        "category": "certificates",
        "isMajor": True,
        "impact": "S/MIME certificates signed on or after 2027-02-01 that carry the emailProtection EKU but no rfc822Name SAN are non-conformant with Apple's root program.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "From 2027-02-01 a subscriber certificate containing id-kp-emailProtection must carry at least one rfc822Name in subjectAltName to conform to Apple Root Program Policy v2.0 §2.3. Certificates signed before that date are unaffected; the trigger is the signing date, not the validity period.",
            "scenario": "The exposure is the identity-only S/MIME profile — certificates issued to a person or device that carry emailProtection because a template always included it, but bind no mailbox. Those stop being conformant on renewal, and the fix is either a real rfc822Name in the SAN or removing the emailProtection EKU from the template. Worth auditing before the renewal wave rather than during it: it is a template change on the CA side, not something a subscriber can patch on the endpoint.",
        },
    },
    {
        "id": "apple-policy-v2-single-trust-purpose",
        "date": "2027-07-01",
        "title": "Apple: Sub-CAs dedicated to one Trust Purpose, Markdown CP/CPS per Trust Purpose, DCR attestation",
        "description": "Apple Root Program Policy v2.0 lands three obligations on 2027-07-01. (1) §1.7: a Subordinate CA Certificate signed on or after 2027-07-01 MUST be dedicated to a single Trust Purpose as defined in Appendix A — the (b) carve-out that ran from 2026-08-01 ends. (2) §1.3.1: all Policy Documents MUST be a combined CP/CPS in Markdown with a .md extension (the Markdown file is authoritative in CCADB), self-contained on the CA's own practices rather than incorporating requirements by reference, and each CP/CPS MUST be scoped to a single Trust Purpose — a CP/CPS may combine only \"Server Authentication\" with \"Legacy TLS\", or \"Secure Email\" with \"Legacy S/MIME\". Trust Purposes and required EKUs must be stated in CP/CPS section 1.4, and a root hierarchy supporting several Trust Purposes needs a distinct document for each. (3) §1.2.3.1: for audit periods starting on or after 2027-07-01 the CA Owner MUST ensure the auditor produces a Detailed Controls Report carrying the four core elements, ensure the auditor contract permits sharing it with Apple, and review it before the attestation letter is finalized. Mozilla sets its own DCR requirement on the same date under MRSP v3.1 — separate root programs, tracked separately.",
        "source": "apple",
        "source_url": "https://github.com/apple/apple-root-program/blob/main/policy.md",
        "category": "audit",
        "isMajor": True,
        "impact": "CA Owners in the Apple program must have their hierarchies split by Trust Purpose, their CP/CPS set rewritten as per-Trust-Purpose Markdown documents, and DCR-capable audit engagements contracted before an audit period starting on or after 2027-07-01.",
        "is_estimated": False,
        "consequences": {
            "enforcement": "Apple Root Program Policy v2.0 requires single-Trust-Purpose dedication for Sub-CA certificates signed on or after 2027-07-01 (§1.7), Markdown CP/CPS documents scoped to one Trust Purpose (§1.3.1), and Detailed Controls Report attestation duties for audit periods starting on or after that date (§1.2.3.1). All three run through the root program's audit and incident machinery rather than a browser-side technical block.",
            "scenario": "No action unless you operate a CA, and the deadline is earlier than it looks for anyone whose audit period starts on 1 July: the DCR duty attaches to the period, so the auditor contract and report format have to be settled before the period opens, not before the report is filed. The CP/CPS work is the heavier half — a multi-purpose hierarchy documented in one PDF becomes several Markdown documents, each describing practices rather than citing requirements, with the Trust Purpose and EKUs declared in section 1.4.",
        },
    },
]

# =============================================================================
# RELATED GUIDES - fixmycert.com guides attached to deadlines
# Default mapping by deadline category; a deadline entry can set its own
# "relatedGuides" list to override (entry value wins, no merge). The frontend
# renders these chips first and only falls back to its keyword heuristic when
# a deadline has none. "hasVideo" marks guides with an embedded video so the
# UI can badge the chip. Paths are verified against the content tracker —
# check ct_get_content before adding a new one; "certificates" is deliberately
# unmapped (too heterogeneous: validity reductions, distrusts, linting).
# =============================================================================

CATEGORY_RELATED_GUIDES = {
    "certificate-transparency": [
        {"title": "Certificate Transparency", "url": "/guides/certificate-transparency", "hasVideo": True},
    ],
    "root-store": [
        {"title": "Root Stores", "url": "/guides/root-stores", "hasVideo": False},
    ],
    "eku": [
        {"title": "ClientAuth EKU Deprecation Guide", "url": "/guides/client-authentication-eku-sunset", "hasVideo": True},
        {"title": "How to Find Client Authentication Certificates", "url": "/guides/find-client-authentication-certificates", "hasVideo": True},
    ],
    "pqc": [
        {"title": "Post-Quantum Cryptography", "url": "/guides/post-quantum-cryptography", "hasVideo": False},
    ],
    "validation": [
        {"title": "Domain Validation Methods", "url": "/guides/domain-validation-methods", "hasVideo": False},
        {"title": "DCV Methods Sunset", "url": "/guides/dcv-methods-sunset", "hasVideo": True},
    ],
    "revocation": [
        {"title": "Revocation", "url": "/guides/revocation", "hasVideo": False},
    ],
    "governance": [
        {"title": "What is a CPS?", "url": "/guides/what-is-a-cps", "hasVideo": False},
    ],
    "algorithm-deprecation": [
        {"title": "Hash Functions", "url": "/guides/hash-functions", "hasVideo": False},
        {"title": "RSA vs ECC", "url": "/guides/rsa-vs-ecc", "hasVideo": False},
    ],
}

_VALIDITY_TIMELINE_GUIDES = [
    {"title": "47-Day Certificate Timeline", "url": "/guides/47-day-certificate-timeline", "hasVideo": True},
]

_MOZILLA_MRSP_GUIDE = {
    "title": "Mozilla Root Store Policy v3.1",
    "url": "/compliance/mozilla-root-store-policy-v3-1",
    "hasVideo": False,
}

# Per-deadline overrides by id, consulted before the category map in
# get_all_deadlines_unified(). Works for both DEADLINES entries and
# framework deadlines. Use when the category default is absent (the
# heterogeneous "certificates" category) or topically wrong for one entry.
EXPLICIT_RELATED_GUIDES = {
    # Validity-reduction entries (category "certificates" has no default)
    "validity-200-days": _VALIDITY_TIMELINE_GUIDES,
    "validity-100-days": _VALIDITY_TIMELINE_GUIDES,
    "validity-47-days": _VALIDITY_TIMELINE_GUIDES,
    "short-lived-cert-threshold-7-days": _VALIDITY_TIMELINE_GUIDES,
    # Categorized root-store (it's a Chrome Root Program change) but the
    # topic is ClientAuth EKU — the EKU guides fit better than Root Stores.
    "chrome-clientauth-leaf-sunset": CATEGORY_RELATED_GUIDES["eku"],
    # Code-signing policy OIDs — the keyword heuristic was picking
    # lifecycle guides for this one.
    "cabf-csc32-reserved-policy-oid": [
        {"title": "Code Signing", "url": "/guides/code-signing", "hasVideo": True},
    ],
    # Mozilla policy deadlines get the dedicated MRSP v3.1 guide (which
    # covers the DCR requirement) on top of / instead of the governance
    # default.
    "mozilla-mrsp-3-1-effective": [
        _MOZILLA_MRSP_GUIDE,
        {"title": "What is a CPS?", "url": "/guides/what-is-a-cps", "hasVideo": False},
    ],
    "mozilla-cpcps-content-compliance-deadline": [
        _MOZILLA_MRSP_GUIDE,
        {"title": "What is a CPS?", "url": "/guides/what-is-a-cps", "hasVideo": False},
    ],
    "mozilla-dcr-audit-periods": [_MOZILLA_MRSP_GUIDE],
    # NSA framework deadline; the nsa resource_link is external, so no
    # framework fallback fires.
    "nspm-12-cnss-governance": [
        {"title": "CNSA 2.0 Certificate Management", "url": "/guides/cnsa-2-certificate-management", "hasVideo": False},
    ],
    # Entrust distrust deadlines — same Root Stores guide the DigiCert
    # distrust cards use (category "certificates" has no default).
    "chrome-entrust-distrust": CATEGORY_RELATED_GUIDES["root-store"],
    "apple-entrust-distrust": CATEGORY_RELATED_GUIDES["root-store"],
    "mozilla-entrust-distrust": CATEGORY_RELATED_GUIDES["root-store"],
    "microsoft-entrust-distrust": CATEGORY_RELATED_GUIDES["root-store"],
    # Code-signing validity reduction.
    "code-signing-validity-460": [
        {"title": "Code Signing", "url": "/guides/code-signing", "hasVideo": True},
    ],
    # Precertificate Signing CA sunset — a CT ecosystem change.
    "sc092-precert-signing-ca-sunset": CATEGORY_RELATED_GUIDES["certificate-transparency"],
    # August 2026 Microsoft root disable. Same Root Stores guide as the other
    # distrust cards ("certificates" has no default, and the keyword heuristic
    # is not reliable here). A dedicated /guides/digicert-global-root-g2 is
    # queued for the 2026-08-18 content batch and is the better target once it
    # ships — it is deliberately NOT listed yet, because it is not registered in
    # the content tracker and the Hub would render a chip to a 404 until then.
    # Swap it in (append, keep Root Stores) after ct_get_content resolves it.
    "microsoft-august-2026-root-disable": CATEGORY_RELATED_GUIDES["root-store"],
}

# =============================================================================
# REGULATORY FRAMEWORKS - DORA, NIS2, UK CSR Bill
# These track regulatory compliance that impacts certificate management
# =============================================================================

REGULATORY_FRAMEWORKS = [
    {
        "framework_id": "dora",
        "name": "DORA",
        "full_name": "Digital Operational Resilience Act",
        "jurisdiction": "eu",
        "effective_date": "2025-01-17",
        "description": "EU regulation requiring financial entities to ensure ICT risk management, incident reporting, resilience testing, and third-party risk management.",
        "applies_to": ["Financial institutions", "Insurance", "Investment firms", "Crypto-asset providers", "ICT service providers to financial sector"],
        "certificate_relevance": "Certificate inventory, CA vendor assessment, incident response for cert outages, resilience testing",
        "resource_link": "/guides/dora-certificate-management",
        "deadlines": [
            {
                "id": "dora-effective",
                "title": "DORA Effective",
                "date": "2025-01-17",
                "status": "ongoing",
                "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554",
                "category": "effective",
                "impact": "All EU financial entities must comply with DORA's ICT risk management, incident reporting, and third-party risk requirements - no transitional period.",
                "isMajor": True,
                "description": "Regulation now applies to all EU financial entities. No transitional period.",
                "consequences": {
                    "enforcement": "DORA applies directly to in-scope EU financial entities from 2025-01-17 with no transitional period. Competent authorities supervise compliance and must be equipped with supervisory, investigatory and sanctioning powers, including remedial measures and administrative penalties (Article 50).",
                    "scenario": "The certificate angle sits in the ICT risk-management and third-party chapters: your public CA is an ICT third-party provider, and an expired certificate that takes a customer-facing service down is an ICT-related incident with reporting obligations attached. Teams that cannot produce a current certificate inventory, name the CAs behind it, and show tested renewal and revocation procedures are the ones that struggle in the first supervisory conversation.",
                },
            },
            {
                "id": "dora-roi-submission",
                "title": "ROI Submission to ESAs",
                "date": "2025-04-30",
                "source_url": "https://www.eba.europa.eu/publications-and-media/press-releases/esas-provide-roadmap-towards-designation-ctpps-under-dora",
                "category": "reporting",
                "impact": "Financial entities must have their Registers of Information on ICT third-party arrangements complete and submitted through national competent authorities.",
                "isMajor": True,
                "description": "National competent authorities submit Registers of Information on ICT third-party arrangements to ESAs.",
                "consequences": {
                    "enforcement": "Under the ESAs' roadmap, competent authorities had to submit financial entities' Registers of Information on ICT third-party arrangements to the ESAs by 2025-04-30 — meaning entities' own national deadlines fell earlier. Those registers are the input for designating critical ICT third-party providers.",
                    "scenario": "The register must name every ICT third-party arrangement, and certificate services are the easiest ones to leave out: the public CA, the certificate lifecycle platform, an HSM-as-a-service provider, an ACME endpoint fronted by a CDN. An incomplete register only reveals itself when a supervisor asks who issues your certificates and the answer is not in the file.",
                },
            },
            {
                "id": "dora-ctpp-designation",
                "title": "CTPP Designation Notifications",
                "date": "2025-07-31",
                "source_url": "https://www.eba.europa.eu/publications-and-media/press-releases/esas-provide-roadmap-towards-designation-ctpps-under-dora",
                "category": "oversight",
                "impact": "Designated CTPPs come under direct ESA oversight; financial entities should verify which of their ICT providers are designated.",
                "isMajor": True,
                "description": "ESAs notify ICT third-party service providers of their classification as Critical Third-Party Providers.",
                "consequences": {
                    "enforcement": "After the register submissions, the ESAs notify ICT third-party providers of their designation as Critical Third-Party Providers, followed by a six-week objection period. Designated CTPPs come under direct ESA oversight through a Lead Overseer.",
                    "scenario": "For a certificate team the practical step is checking whether anything in your certificate chain — the CA, the CLM vendor, the cloud platform hosting your PKI — appears on the designated list. Designation does not transfer any of your own obligations, but it changes the leverage in the vendor conversation; where a provider is not designated, your contract terms and exit plan are doing all the work.",
                },
            },
            {
                "id": "dora-ec-review",
                "title": "EC Review Report Due",
                "date": "2026-01-17",
                "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554",
                "category": "review",
                "impact": "No direct obligations - the Commission's Article 58 report may propose scope expansion, so watch for follow-on requirements.",
                "isMajor": False,
                "description": "European Commission reviews DORA implementation and submits report to Parliament on potential scope expansion (Article 58)."
            },
            {
                "id": "dora-roi-2026",
                "title": "Annual ROI Submission (2026)",
                "date": "2026-03-31",
                "source_url": "https://eba.europa.eu/activities/direct-supervision-and-oversight/digital-operational-resilience-act/preparation-dora-application",
                "category": "reporting",
                "impact": "Annual RoI submission is now a recurring 31 March obligation; regulators expect more mature ICT third-party documentation than the first cycle.",
                "isMajor": True,
                "description": "Second annual Register of Information submission. From 2026 onwards, competent authorities submit RoIs to the ESAs by 31 March each year (per EBA DORA reporting FAQ). Regulators expect more mature submissions with detailed ICT third-party documentation.",
                "consequences": {
                    "enforcement": "From 2026 onwards competent authorities submit Registers of Information to the ESAs by 31 March each year, with 31 December of the preceding year as the reference date. Each competent authority sets an earlier national deadline for financial entities to report to it.",
                    "scenario": "The second cycle is read against a higher bar than the first, and the 31 December reference date is what catches people out: the register has to reflect arrangements as they stood at year end, not as they stand when someone assembles the file in March. Switch CAs in January and the old provider still has to appear, in its December state.",
                },
            },
            {
                "id": "dora-ctpp-eu-subsidiary-window",
                "title": "Non-EU CTPPs: EU subsidiary window closes (Art. 31(12))",
                "date": "2026-11-18",
                "source_url": "https://www.eba.europa.eu/publications-and-media/press-releases/european-supervisory-authorities-designate-critical-ict-third-party-providers-under-digital",
                "category": "requirements",
                "impact": "Financial entities using third-country CTPPs without an EU subsidiary must be prepared to exit those arrangements if the provider fails to establish one.",
                "isMajor": False,
                "description": "DORA Art. 31(12): financial entities may only use a third-country ICT provider designated as critical if it has established an EU subsidiary within 12 months of designation. The ESAs designated the first 19 CTPPs on 18 November 2025, so the window closes ~18 November 2026 (date derived from the 12-month statutory period). Most designated groups already operate EU subsidiaries; the practical exposure is for financial entities using third-country-established designees (e.g. Bloomberg L.P., FIS, IBM Corp., Kyndryl, NTT DATA, TCS, and UK-established Colt and LSEG entities) if any fail to maintain one.",
                "is_estimated": True
            }
        ]
    },
    {
        "framework_id": "nis2",
        "name": "NIS2",
        "full_name": "Network and Information Security Directive 2",
        "jurisdiction": "eu",
        "effective_date": "2024-10-17",
        "description": "EU directive expanding cybersecurity requirements to essential and important entities across multiple sectors.",
        "applies_to": ["Energy", "Transport", "Banking", "Health", "Digital infrastructure", "ICT service management", "Public administration"],
        "certificate_relevance": "Encryption requirements, supply chain security, incident notification for certificate outages",
        "resource_link": "/guides/nis2-certificate-management",
        "deadlines": [
            {
                "id": "nis2-transposition",
                "title": "Transposition Deadline",
                "date": "2024-10-17",
                "source_url": "https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng",
                "category": "effective",
                "impact": "NIS2 obligations begin applying as member states transpose; entities in scope must track their national implementation timelines.",
                "isMajor": True,
                "description": "Member states required to transpose NIS2 into national law. 23 states missed deadline; EC opened infringement proceedings.",
                "jurisdiction_detail": "EU",
                "consequences": {
                    "enforcement": "Member states were required to transpose NIS2 into national law by 2024-10-17. Obligations bite through national law, and the directive sets maximum administrative fines of at least EUR 10 million or 2% of total worldwide annual turnover (whichever is higher) for essential entities, and at least EUR 7 million or 1.4% for important entities.",
                    "scenario": "Because 23 member states missed the deadline, the real problem is jurisdictional rather than technical. A group operating across the EU faces different in-force dates, different registration portals and different national deadlines for the same directive, so a single group-wide compliance date does not exist. Certificate-relevant duties — encryption, supply-chain security, incident notification when an expired certificate causes an outage — have to be tracked per country, per entity.",
                },
            },
            {
                "id": "nis2-netherlands-cbw",
                "title": "Netherlands Cyberbeveiligingswet Effective",
                "date": "2026-08-15",
                # type (b) in-force regime, same as the other national NIS2
                # entries. Set on 2026-08-15, its commencement date: it could
                # not carry "ongoing" while future-dated (the date-consistency
                # test requires future entries to compute "upcoming"), and
                # from 2026-08-16 it would otherwise render green "Completed".
                "status": "ongoing",
                "source_url": "https://zoek.officielebekendmakingen.nl/stb-2026-189.html",
                "category": "national",
                "impact": "Dutch entities in scope must register with the NCSC via mijn.ncsc.nl, meet the duty of care for network and information system security, and report significant incidents to their CSIRT within the statutory deadlines.",
                "isMajor": True,
                "description": "The Cyberbeveiligingswet (Cbw), the Dutch NIS2 transposition, enters into force on 15 August 2026 together with the Wet weerbaarheid kritieke entiteiten (CER transposition). Commencement is set by the Cyberbeveiligingsbesluit — Besluit van 8 juli 2026, Staatsblad 2026, 189, published 10 July 2026 — which also carries the implementing rules. Around 8,000 Dutch organizations come into scope, and roughly 500 are formally designated as critical entities under the Wwke.",
                "jurisdiction_detail": "Netherlands",
                "consequences": {
                    "enforcement": "From 15 August 2026 in-scope entities must register with the NCSC through mijn.ncsc.nl, take measures to manage the risks to the security of their network and information systems (duty of care), and report significant incidents to their CSIRT within the statutory timeframes. Registration is mandatory as of that date.",
                    "scenario": "The Netherlands lands late in the NIS2 wave, so the trap for a group already compliant elsewhere is assuming the Dutch entity inherits that work — it does not: registration is per-entity, through a Dutch portal, on a Dutch clock. Scope is self-assessed, so nobody writes to tell a mid-sized Dutch subsidiary it now has duties. For a certificate team the concrete item is the incident-reporting path: an expired certificate that takes a regulated service offline is a reportable significant incident, and the reporting deadline runs from detection, not from the post-mortem.",
                },
            },
            {
                "id": "nis2-czechia-effective",
                "title": "Czechia Cybersecurity Act Effective",
                "date": "2025-11-01",
                "status": "ongoing",
                "source_url": "https://portal.nukib.gov.cz/informacni-servis/aktualne/6904ea6c0fc0983fc00a08e2",
                "category": "national",
                "impact": "Czech in-scope entities must notify their regulated service to NÚKIB through the NÚKIB Portal within 60 days of meeting the conditions, then meet the duties of whichever obligation regime they land in.",
                "isMajor": True,
                "description": "Act No. 264/2025 Sb. on cybersecurity entered into force on 1 November 2025 — a full recodification transposing NIS2 and replacing Act No. 181/2014 Sb. Providers are split into a higher-obligations and a lower-obligations regime (§ 8), covering an estimated 6,000+ regulated organizations. NÚKIB is the supervisory authority and its Portal is the mandatory channel between regulated entities and the authority.",
                "jurisdiction_detail": "Czechia",
                "consequences": {
                    "enforcement": "Section 6(1) of Act No. 264/2025 Sb. requires a provider meeting the registration conditions to notify its regulated service to NÚKIB within 60 days of the day those conditions were met — the clock runs per entity from the date it comes into scope, not once from the Act's commencement. Notification goes through the NÚKIB Portal, and changes that could move a provider between obligation regimes must likewise be reported within 60 days (§ 9).",
                    "scenario": "The 60-day clock is what catches people, because it is not a date anyone has in a calendar — it starts when the organization crosses into scope, which can happen through headcount growth, an acquisition or a new service line. A Czech subsidiary comfortably outside Act 181/2014 can become a regulated provider under the new Act without anything changing in its network. For a certificate team the real work starts after registration: the regime you land in decides how much of your cryptography, renewal and incident-reporting practice has to be documented rather than merely working.",
                },
            },
            {
                "id": "nis2-germany-bsi",
                "title": "Germany BSI Act Effective",
                "date": "2025-12-06",
                "status": "ongoing",
                "source_url": "https://www.recht.bund.de/bgbl/1/2025/301/VO.html",
                "category": "national",
                "impact": "Entities in scope in Germany become subject to the amended BSI Act's registration, security, and reporting obligations.",
                "isMajor": True,
                "description": "German NIS2 implementation via amended BSI Act enters into force.",
                "jurisdiction_detail": "Germany",
                "consequences": {
                    "enforcement": "Germany's NIS2 implementation act (BGBl. I 2025 Nr. 301, promulgated 2025-12-05) brings the amended BSI Act into force, making registration, risk-management and incident-reporting duties applicable to in-scope German entities under BSI supervision.",
                    "scenario": "Scope is self-assessed, which is where German entities get caught: nobody writes to tell you that you are in scope. A mid-sized manufacturer or logistics operator that clears the size and sector thresholds is subject to the duties whether or not it has worked that out, and the three-month registration clock runs from the date it became in scope — not from the date someone noticed.",
                },
            },
            {
                "id": "nis2-sweden-effective",
                "title": "Sweden Cybersecurity Act Effective",
                "date": "2026-01-15",
                "status": "ongoing",
                "source_url": "https://svenskforfattningssamling.se/doc/20251506.html",
                "category": "national",
                "impact": "Swedish in-scope entities are bound by the Cybersecurity Act's risk-management and incident-reporting duties from 15 January 2026, whether or not their registration is complete.",
                "isMajor": True,
                "description": "Cybersäkerhetslagen (SFS 2025:1506), Sweden's NIS2 transposition — adopted by the Riksdag 10 December 2025, issued 11 December 2025, in force 15 January 2026. It repeals lagen (2018:1174) om informationssäkerhet för samhällsviktiga och digitala tjänster, which continues to govern breaches committed before commencement. The companion cybersäkerhetsförordningen (2025:1507) designates the authorities: supervision is sectoral (Finansinspektionen for banking and financial market infrastructure, Statens energimyndighet for energy, Transportstyrelsen for transport, and others under § 7), while Försvarets radioanstalt is the single point of contact (§ 23), the CSIRT unit (§ 31) and the cyber crisis management authority (§ 35).",
                "jurisdiction_detail": "Sweden",
                "consequences": {
                    "enforcement": "The transitional provisions to SFS 2025:1506 bring the Act into force on 15 January 2026 and repeal lagen (2018:1174), with the repealed law still applying to breaches committed before that date. The Chapter 2 duties on operators — risk management, notification and incident reporting — therefore bind in-scope entities from 15 January 2026; incident reports go to the CSIRT unit and notifications to the single point of contact under the companion ordinance. Registration is not a precondition for the duties, so being unregistered is not a defence.",
                    "scenario": "The old Swedish law covered a much narrower population, so the trap is an entity that sat outside lagen (2018:1174) assuming continuity. The obligations do not wait for anyone's registration to be processed: from 15 January 2026 an incident that degrades a service — including an expired certificate taking a public-facing system offline — is reportable on the NIS2 timescale, and \"our registration was still in progress\" changes nothing. Treat the in-force date, not your registration date, as the point where evidence starts to matter. Sweden's split supervision adds a second problem: a group spanning banking and energy answers to different supervisory authorities for different entities.",
                },
            },
            {
                "id": "nis2-eu-cyclone",
                "title": "EU-CyCLONe Report",
                "date": "2026-01-17",
                "source_url": "https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng",
                "category": "reporting",
                "impact": "Informational - the EU-CyCLONe report to Parliament carries no direct obligations for regulated entities.",
                "isMajor": False,
                "description": "European cyber crisis liaison network submits report to Parliament and Council.",
                "jurisdiction_detail": "EU"
            },
            {
                "id": "nis2-germany-registration",
                "title": "Germany Registration Deadline",
                "date": "2026-04-06",
                "source_url": "https://www.bsi.bund.de/DE/Themen/Regulierte-Wirtschaft/NIS-2-regulierte-Unternehmen/NIS-2-Pflichten/nis-2-pflichten_node.html",
                "category": "registration",
                "impact": "In-scope entities must complete BSI registration; missing the three-month window is itself a violation.",
                "isMajor": True,
                "description": "Entities must register with Federal Office for Information Security (BSI) within 3 months of BSI Act entering force.",
                "jurisdiction_detail": "Germany",
                "consequences": {
                    "enforcement": "In-scope entities must register with the BSI within three months of first becoming subject to NIS2, through a two-step process — Mein Unternehmenskonto, then the BSI portal — submitting name, address, legal form, sector, contacts, headcount and turnover.",
                    "scenario": "Registration is the cheapest obligation in the regime and the easiest to miss, because it is an administrative task with no natural technical owner. It falls between IT, legal and the company secretary until the window has already closed — and being unregistered is a standalone, visible breach regardless of how good the underlying security actually is.",
                },
            },
            {
                "id": "nis2-italy-audit",
                "title": "Italy Annual Categorization Deadline",
                "date": "2026-06-30",
                "status": "ongoing",
                "source_url": "https://www.acn.gov.it/portale/nis/categorizzazione",
                "category": "reporting",
                "impact": "Italian in-scope entities must file and keep current their categorized list of activities and services on the ACN platform before the window closes each year; ACN may challenge the categories afterwards.",
                "isMajor": True,
                "description": "Close of the annual window (1 May - 30 June) in which essential and important entities must communicate and update, via the ACN digital platform, the list of their activities and services with the relevant impact categories (Article 30(1), D.Lgs. 138/2024). The obligation was introduced by ACN Determination 155238 of 20 April 2026.",
                "jurisdiction_detail": "Italy",
                "consequences": {
                    "enforcement": "Article 30(1) of the Italian NIS decree requires essential and important entities to communicate and update their list of activities and services, with relevance categories, on the ACN platform between 1 May and 30 June each year. ACN may then verify submissions on a sample basis, including by comparison against similar entities.",
                    "scenario": "The categorization gets treated as a form-filling exercise and done in late June by whoever holds the portal login. Because ACN can challenge it afterwards, the exposure is a category you cannot justify — keep the criteria, sources and people involved on record. The same file is what you reuse for next year's window and for any ACN follow-up.",
                },
            },
            {
                "id": "nis2-luxembourg-registration",
                "title": "Luxembourg Self-Registration Deadline",
                "date": "2026-07-10",
                "source_url": "https://legilux.public.lu/eli/etat/leg/loi/2026/05/05/a225/jo",
                "category": "registration",
                "impact": "In-scope Luxembourg entities must self-register with the ILR; non-registration is itself a sanctionable breach.",
                "isMajor": True,
                "description": "Entities in scope of Luxembourg's NIS2 law (in force 10 May 2026) must self-register with the ILR by 10 July 2026 (Article 11). Non-registration is itself a sanctionable breach.",
                "jurisdiction_detail": "Luxembourg",
                "consequences": {
                    "enforcement": "Article 11 of the Luxembourg law of 5 May 2026 (in force 10 May 2026) requires in-scope entities to self-register with the ILR within two months of entry into force — by 2026-07-10 — providing name, contact details, sector and sub-sector, the member states where they provide services, and their size.",
                    "scenario": "Self-registration means the ILR will not come and find you; if the entity clears the thresholds and nobody files, the failure stays silent until it is a sanction. Luxembourg's fund-administration and ICT-services population is full of entities assuming a group filing elsewhere covers them. It does not — registration is per entity, per member state.",
                },
            },
            {
                "id": "nis2-austria-effective",
                "title": "Austria Full Application",
                "date": "2026-10-01",
                "source_url": "https://www.ris.bka.gv.at/Dokumente/Bundesnormen/NOR40273912/NOR40273912.html",
                "category": "effective",
                "impact": "Austrian in-scope entities must fully comply with the Network and Information System Security Act 2026, including security measures and reporting duties.",
                "isMajor": True,
                "description": "Network and Information System Security Act 2026 fully applicable.",
                "jurisdiction_detail": "Austria",
                "consequences": {
                    "enforcement": "The Netz- und Informationssystemsicherheitsgesetz 2026 (NISG 2026) becomes fully applicable on 2026-10-01, when risk-management and incident-reporting duties start applying to roughly 4,000 in-scope Austrian entities. Registration follows within three months, and a self-declaration on implemented risk-management measures within twelve months.",
                    "scenario": "The twelve-month self-declaration is what should shape the work now: whatever measures you declare in autumn 2027 have to actually exist. For a certificate team that means a real inventory, named renewal owners, and evidence that revocation has been exercised rather than merely documented. Entities treating 1 October as the start of a planning phase rather than a compliance date run out of runway.",
                },
            },
            {
                "id": "nis2-italy-security",
                "title": "Italy Security Requirements",
                "date": "2026-10-31",
                "source_url": "https://www.acn.gov.it/portale/en/nis/modalita-specifiche-base",
                "category": "requirements",
                "impact": "Italian in-scope entities must implement the minimum security requirements from the ACN technical annexes by this date.",
                "isMajor": True,
                "description": "ACN technical annexes establishing minimum security requirements become effective (ACN states October 2026; 18 months from April 2025 list consolidation).",
                "jurisdiction_detail": "Italy",
                "is_estimated": True,
                "consequences": {
                    "enforcement": "ACN Determination 379907/2025 sets the base security measures: important entities implement Annex 1 and essential entities Annex 2 within 18 months of notice of inclusion on the national NIS list, with ACN stating October 2026 as the implementation deadline.",
                    "scenario": "The annexes are explicit about cryptography and certificate handling, so this is the point where an Italian in-scope entity has to show more than intent: an inventory of certificates, named owners, and renewal and revocation procedures somebody has actually run. Eighteen months sounds generous until you reach the certificates sitting on appliances nobody has admin credentials for.",
                },
            },
            {
                "id": "nis2-eu-cyclone-2027",
                "title": "EU-CyCLONe Report (2027)",
                "date": "2027-07-17",
                "source_url": "https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng",
                "category": "reporting",
                "impact": "Informational - recurring EU-CyCLONe reporting cycle with no direct obligations for regulated entities.",
                "isMajor": False,
                "description": "Next 18-month cycle report from cyber crisis liaison network.",
                "jurisdiction_detail": "EU"
            }
        ]
    },
    {
        "framework_id": "uk-csr",
        "name": "UK CSR Bill",
        "full_name": "Cyber Security and Resilience Bill",
        "jurisdiction": "uk",
        "effective_date": None,
        "description": "UK legislation to strengthen cyber resilience of critical infrastructure and digital services, expanding NIS regulations.",
        "applies_to": ["Critical national infrastructure", "Digital service providers", "Managed service providers"],
        "certificate_relevance": "Expected to include requirements similar to DORA for certificate management and ICT resilience",
        "resource_link": "/guides/uk-csr-bill-certificate-management",
        "deadlines": [
            {
                "id": "uk-csr-introduced",
                "title": "Bill Introduced",
                "date": "2025-11-12",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "No obligations yet - legislative milestone only; organizations likely in scope should begin tracking the bill.",
                "isMajor": False,
                "description": "Cyber Security and Resilience (Network and Information Systems) Bill introduced to House of Commons."
            },
            {
                "id": "uk-csr-second-reading",
                "title": "Second Reading Passed",
                "date": "2026-01-06",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "No obligations yet - legislative milestone only.",
                "isMajor": False,
                "description": "Bill passed second reading in House of Commons."
            },
            {
                "id": "uk-csr-committee-begins",
                "title": "Committee Stage Begins",
                "date": "2026-02-03",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "No obligations yet - committee scrutiny may amend the bill's scope and duties.",
                "isMajor": False,
                "description": "Public Bill Committee begins line-by-line scrutiny. Oral evidence sessions."
            },
            {
                "id": "uk-csr-committee-reports",
                "title": "Committee Stage Reports",
                "date": "2026-03-05",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "Committee amendments settle the bill's shape - a good point to reassess whether your organization falls in scope.",
                "isMajor": False,
                "description": "Public Bill Committee expected to report by 5:00pm."
            },
            {
                "id": "uk-csr-lords-stage",
                "title": "Bill Enters House of Lords",
                "date": "2026-06-17",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "No obligations yet - legislative milestone only; Royal Assent moves closer.",
                "isMajor": False,
                "description": "Bill passed the House of Commons (third reading 16 Jun 2026) and had its first reading in the House of Lords on 17 June 2026. Lords second reading completed 14 July 2026."
            },
            {
                "id": "uk-csr-royal-assent",
                "title": "Royal Assent (Estimated)",
                "date": "2026-12-31",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "Once the bill becomes law, secondary legislation and enforcement timelines start; in-scope organizations should begin gap assessments.",
                "isMajor": True,
                "description": "Bill was carried over and reintroduced May 2026; passed Commons third reading 16 Jun 2026; Lords second reading completed 14 Jul 2026. The bill now proceeds to Committee stage in the Lords, where a transnational repression amendment is expected from Lord Alton of Liverpool - it would bar sharing private information with overseas authorities in jurisdictions that cannot guarantee the right to a fair trial, and was voted down in the Commons. Legislative stage only; no new date-certain. Royal Assent still expected late 2026.",
                "is_estimated": True,
                "consequences": {
                    "enforcement": "Royal Assent makes the bill law but creates no immediate duties. The bill amends the Network and Information Systems Regulations 2018 to widen scope and update incident-reporting duties, and confers powers on the Secretary of State; substantive requirements follow in secondary legislation. As of July 2026 the bill is in the Lords, with committee stage scheduled for 1 September 2026 and no Royal Assent date set.",
                    "scenario": "Nothing to comply with on the day. The useful work while the bill is in the Lords is a scope check — managed service providers and data centre operators are the newly captured populations — and a gap assessment against the existing NIS Regulations, because the phased duties arriving through 2027 will assume a certificate inventory and an incident-reporting capability that you either have or do not.",
                },
            },
            {
                "id": "uk-csr-implementation",
                "title": "Implementation Begins (Estimated)",
                "date": "2027-12-31",
                "source_url": "https://www.gov.uk/government/publications/cyber-security-and-resilience-network-and-information-systems-bill-factsheets",
                "category": "implementation",
                "impact": "Phased duties begin landing via secondary legislation; UK critical infrastructure operators and digital/managed service providers should be ready to comply as requirements are laid.",
                "isMajor": True,
                "description": "Phased implementation via secondary legislation expected during 2027, following Royal Assent. Detailed requirements to follow.",
                "is_estimated": True
            }
        ]
    },
    {
        "framework_id": "cabforum",
        "name": "CA/Browser Forum",
        "full_name": "Certificate Authority/Browser Forum",
        "jurisdiction": "global",
        "effective_date": "2011-06-22",
        "description": "Industry consortium defining certificate issuance and management standards for publicly-trusted TLS, code signing, and S/MIME certificates.",
        "applies_to": ["Certificate Authorities", "Web browsers", "TLS certificate subscribers", "Code signing entities", "S/MIME users"],
        "certificate_relevance": "Primary source of certificate issuance requirements, validation rules, and operational requirements",
        "resource_link": "https://cabforum.org",
        "deadlines": []
    },
    {
        "framework_id": "nist",
        "name": "NIST",
        "full_name": "National Institute of Standards and Technology",
        "jurisdiction": "us",
        "effective_date": "2024-08-13",
        "description": "US government agency setting cryptographic standards including post-quantum cryptography (FIPS 203/204/205), key management (SP 800-57), and algorithm transitions (SP 800-131A, IR 8547).",
        "applies_to": ["US federal agencies", "Government contractors", "Organizations following NIST guidance", "Critical infrastructure"],
        "certificate_relevance": "Cryptographic algorithm requirements, key sizes, PQC transition timeline, certificate validity guidance",
        "resource_link": "https://csrc.nist.gov/projects/post-quantum-cryptography",
        "deadlines": []
    },
    {
        "framework_id": "nsa",
        "name": "NSA CNSA 2.0",
        "full_name": "NSA Commercial National Security Algorithm Suite 2.0",
        "jurisdiction": "us",
        "effective_date": "2022-09-01",
        "description": "NSA guidance for transitioning National Security Systems to quantum-resistant cryptography. Defines approved PQC algorithms (ML-KEM, ML-DSA) and transition timeline through 2035.",
        "applies_to": ["National Security Systems", "Defense contractors", "Intelligence community", "Critical infrastructure handling classified data"],
        "certificate_relevance": "PQC algorithm requirements for certificates, key exchange, and signatures. Defines timeline for exclusive PQC usage.",
        "resource_link": "https://www.nsa.gov/Cybersecurity/Post-Quantum-Cybersecurity-Resources/",
        "deadlines": [
            {
                "id": "nspm-12-cnss-governance",
                "title": "NSPM-12 — NSS cybersecurity governance overhauled",
                "date": "2026-06-12",
                "source_url": "https://www.whitehouse.gov/presidential-actions/2026/06/national-security-presidential-memorandum-nspm-12/",
                "category": "regulatory",
                "impact": "Governance change for National Security Systems - no new certificate requirements; CNSA and CNSSP 15 cryptographic requirements continue unchanged.",
                "isMajor": False,
                "description": "NSPM-12 (signed June 12, 2026) rescinds NSD-42 (1990) and NSM-8 (2022), re-establishes the Committee on National Security Systems (CNSS) with binding directive authority, and designates the Director of NSA as National Manager for National Security Systems. Cryptographic requirements for NSS continue to flow through CNSSP 15, the policy vehicle for CNSA."
            }
        ]
    }
]

# CA/Browser Forum and other document versions
# UPDATE THESE when new versions are released
# 
# To check for updates:
#   - TLS BR: https://cabforum.org/working-groups/server/baseline-requirements/documents/
#   - EV: https://cabforum.org/working-groups/server/extended-validation/documents/
#   - Code Signing: https://cabforum.org/working-groups/code-signing/documents/
#   - S/MIME: https://cabforum.org/working-groups/smime/documents/
#   - NetSec: https://cabforum.org/working-groups/netsec/documents/
#
CABF_DOCUMENTS = [
    {
        "id": "tls-br",
        "name": "TLS Baseline Requirements",
        "version": "2.2.9",
        "date": "Aug 2026",
        "url": "https://cabforum.org/working-groups/server/baseline-requirements/documents/",
    },
    {
        "id": "ev-guidelines",
        "name": "EV Guidelines",
        "version": "2.0.3",
        "date": "Jul 2026",
        "url": "https://cabforum.org/working-groups/server/extended-validation/documents/",
    },
    {
        "id": "code-signing-br",
        "name": "Code Signing BRs",
        "version": "3.11",
        "date": "Jun 2026",
        "url": "https://cabforum.org/working-groups/code-signing/documents/",
    },
    {
        "id": "smime-br",
        "name": "S/MIME BRs",
        "version": "1.0.15",
        "date": "Jul 2026",
        "url": "https://cabforum.org/working-groups/smime/documents/",
    },
    {
        "id": "netsec",
        "name": "Network Security Reqs",
        "version": "2.0.5",
        "date": "Jul 2025",
        "url": "https://cabforum.org/working-groups/netsec/documents/",
    },
]

# =============================================================================
# ROOT STORE POLICIES - Comprehensive comparison data
# =============================================================================

ROOT_STORES = [
    {
        "id": "chrome",
        "name": "Chrome Root Program",
        "version": "1.8",
        "url": "https://www.chromium.org/Home/chromium-security/root-ca-policy/",
        "platforms": ["Chrome", "Chromium-based browsers (Edge, Brave, Opera)", "Android"],
        "keyRequirements": [
            "Certificate Transparency (SCTs) mandatory",
            "CT pre-logging required for all precertificates before issuance (effective June 15, 2026)",
            "Multi-purpose roots being phased out",
            "Dedicated TLS hierarchies required",
            "Root consolidation: CAs with >2 roots must submit consolidation plan by June 15, 2026",
            "CRLSets for revocation (no real-time OCSP)",
            "ClientAuth EKU sunset in progress (enforcement June 2026)",
            "CTLM (Certificate Transparency Log Monitor) now included in monthly Windows CTL"
        ],
        "recentActions": [
            {"action": "Entrust distrust", "date": "2024-11-11", "description": "Chrome 131 distrusts Entrust roots for TLS"},
            {"action": "ClientAuth EKU phase-out", "date": "2025-06-15", "description": "Chrome stops trusting ICAs with mixed ServerAuth+ClientAuth EKUs"}
        ],
        "ekuPolicy": {
            "status": "sunset_in_progress",
            "serverAuthRequired": True,
            "clientAuthProhibited": "2026-06-15",
            "mixedEkuIcaDistrust": "2025-06-15",
            "description": "Public TLS certificates must not include ClientAuth EKU. mTLS must use private PKI."
        },
        "evDisplay": "Hidden since Chrome 77 (Sept 2019)",
        "revocationMethod": "CRLSets",
        "ctRequired": True,
        "automationRequired": "Required for all subordinate CAs by March 15, 2027"
    },
    {
        "id": "mozilla",
        "name": "Mozilla Root Store Policy",
        "version": "3.1",
        "url": "https://wiki.mozilla.org/CA/Root_Store_Policy",
        "platforms": ["Firefox", "Thunderbird"],
        "keyRequirements": [
            "Certificate Transparency required",
            "CRLite for revocation (no real-time OCSP)",
            "CCADB disclosure requirements",
            "Mass revocation capability required by Sept 2025",
            "Dual-purpose root transition plans due April 15, 2026",
            "Full migration to dedicated TLS/S/MIME hierarchies by December 31, 2028",
            "Parked key monitoring with SHA-256 hash disclosure required"
        ],
        "recentActions": [
            {"action": "Entrust distrust", "date": "2024-11-30", "description": "Firefox distrusts Entrust roots for TLS"},
            {"action": "Mass revocation assessment", "date": "2025-06-01", "description": "Third-party assessment required"}
        ],
        "evDisplay": "Shows in certificate viewer",
        "revocationMethod": "CRLite",
        "ctRequired": True,
        "automationRequired": "Disclosure required by April 2025"
    },
    {
        "id": "apple",
        "name": "Apple Root Certificate Program",
        # Policy v2.0, effective 2026-08-01, read from the repo's policy.md
        # header on 2026-08-15. The card said "2024" until then.
        "version": "2.0 (Effective 2026-08-01)",
        "url": "https://github.com/apple/apple-root-program",
        "platforms": ["Safari", "iOS", "iPadOS", "macOS"],
        "keyRequirements": [
            "Certificate Transparency required",
            "OCSP checking enabled",
            "ARI implementation disclosure by April 2025",
            "Initiated 398-day max validity (2020)"
        ],
        "recentActions": [
            {"action": "Entrust distrust", "date": "2024-11-15", "description": "Safari/iOS distrusts Entrust for TLS, S/MIME, timestamping"},
            {"action": "Root Program Policy v2.0", "date": "2026-08-01", "description": "Trust Purposes defined (Appendix A); Sub-CA certificates signed on/after 2026-08-01 need an EKU and must not assert anyExtendedKeyUsage; Apple approval required before issuing to an Externally Operated Sub-CA; new root submissions must be RSA 4096-bit or ECDSA 384-bit minimum"}
        ],
        "evDisplay": "Shows organization in certificate details",
        "revocationMethod": "OCSP",
        "ctRequired": True,
        "automationRequired": "ARI disclosure required"
    },
    {
        "id": "microsoft",
        "name": "Microsoft Trusted Root Program",
        "version": "1.2 (May 2026)",
        "url": "https://github.com/TrustedRootProgram/Program-Requirements",
        "platforms": ["Windows", "Edge (legacy)", "Internet Explorer"],
        "keyRequirements": [
            "Broader root store (~400+ roots)",
            "Includes enterprise/government use cases",
            "Traditional CRL/OCSP checking",
            "Code signing roots included",
            "Single-purpose (EKU-separated) roots for submissions on/after Jul 2026",
            "New roots capped at 10-year validity",
            "Mandatory public incident reporting (Bugzilla, CCADB format)"
        ],
        "recentActions": [
            {"action": "Entrust distrust", "date": "2025-04-16", "description": "Announced Feb 25, 2025. Entrust distrusted for TLS certs issued after this date"},
            {"action": "April 2026 CTL NotBefore distrusts", "date": "2026-05-19", "description": "New issuance distrusted for SwissSign Silver CA - G2, SecureSign RootCA11/CA12, ANCERT, Byte; S/MIME-only NotBefore for 19 legacy roots incl. GoDaddy Class 2/G2, HARICA 2015"},
            {"action": "Requirements v1.2", "date": "2026-05-20", "description": "Single-purpose roots + 10-year max validity for new submissions (effective Jul 1, 2026), suspect-code definition, mandatory Bugzilla/CCADB incident reporting"},
            {"action": "PQC TLS Pilot Program V1.0", "date": "2026-06-08", "description": "ML-DSA-87 pilot roots for closed-environment TLS testing; not publicly trusted"},
            {"action": "June 2026 CTL release", "date": "2026-06-30", "description": "~40 root additions; CTLM policy ships in monthly CTL (opt-in CT validation under test); CTL signatures now SHA-2 only"}
        ],
        "evDisplay": "Shows organization in address bar",
        "revocationMethod": "CRL/OCSP",
        "ctRequired": False,
        "automationRequired": "Not required",
        "notes": "Requirements v1.1/v1.2 (Nov 2025 / May 2026) mandate CA/B Forum BR and CCADB compliance with no exceptions; CT validation is being tested on Windows via the CTLM policy but is not yet required"
    }
]

ROOT_STORE_COMPARISON = {
    "quickComparison": {
        "entrustDistrust": {
            "Chrome": "Nov 11, 2024",
            "Mozilla": "Nov 30, 2024",
            "Apple": "Nov 15, 2024",
            "Microsoft": "April 16, 2025"
        },
        "ctRequired": {
            "Chrome": "Yes (SCTs required)",
            "Mozilla": "Yes",
            "Apple": "Yes",
            "Microsoft": "Not enforced"
        },
        "revocationMethod": {
            "Chrome": "CRLSets (no real-time OCSP)",
            "Mozilla": "CRLite (no real-time OCSP)",
            "Apple": "OCSP",
            "Microsoft": "CRL/OCSP"
        },
        "evDisplay": {
            "Chrome": "Hidden since 2019",
            "Mozilla": "Certificate viewer only",
            "Apple": "Certificate details",
            "Microsoft": "Address bar"
        },
        "multiPurposePhaseOut": {
            "Chrome": "June 2026",
            "Mozilla": "Dec 2028",
            "Apple": "April 2024 (new applicants)",
            "Microsoft": "No timeline"
        },
        "automationRequired": {
            "Chrome": "Encouraged",
            "Mozilla": "Disclosure required",
            "Apple": "ARI disclosure",
            "Microsoft": "Not required"
        },
        "dualPurposeRootDeadline": {
            "Chrome": "June 2026 (new subs serverAuth only)",
            "Mozilla": "Plan due Apr 2026, migrate by Dec 2028",
            "Apple": "Required for new applicants",
            "Microsoft": "No timeline"
        },
        "ctPreLogging": {
            "Chrome": "Required June 15, 2026",
            "Mozilla": "Required",
            "Apple": "Required",
            "Microsoft": "CTLM published, enforcement testing"
        }
    },
    "importantDifferences": [
        {
            "topic": "OCSP Checking",
            "impact": "MEDIUM",
            "description": "Chrome and Mozilla do NOT check OCSP in real-time. They use CRLSets/CRLite instead. A revoked certificate may still work in these browsers until the next update. Always test revocation in multiple browsers."
        },
        {
            "topic": "EV Certificate Display",
            "impact": "INFO",
            "description": "Chrome removed the green EV indicator in 2019. If your organization paid for EV specifically for the visual indicator, it only shows in Edge, Firefox (cert viewer), and Safari now."
        },
        {
            "topic": "Microsoft CT Enforcement Coming",
            "impact": "MEDIUM",
            "description": "Microsoft now publishes CTLM (Certificate Transparency Log Monitor) in monthly Windows CTL. CT validation testing via event logging before individual applications can opt in to enforcement. Windows is moving toward CT enforcement."
        },
        {
            "topic": "Microsoft Secure Boot Certificate Expiration",
            "impact": "HIGH",
            "description": "Original 2011 Secure Boot certificates expire late June 2026. Microsoft rolling out replacement certificates via Windows Update. Devices that miss the update will still boot but fall behind on boot-level security mitigations."
        }
    ]
}

# =============================================================================
# CA ACQUISITIONS & TRANSITIONS
# Track major CA ownership changes and customer migration timelines
# Last verified: 2025-12-29
# =============================================================================

CA_ACQUISITIONS = [
    {
        "id": "sectigo-entrust-2025",
        "acquirer": "Sectigo",
        "acquired": "Entrust Public Certificate Business",
        "status": "completed",
        "summary": "Sectigo acquired Entrust's public certificate business following browser distrust actions. All public TLS, S/MIME, code signing, and document signing certificates affected. Private CA services NOT included.",
        "timeline": [
            {
                "date": "2024-06-27",
                "event": "Google announces Chrome will distrust Entrust",
                "type": "trigger",
                "description": "Following years of compliance failures, Google announces distrust effective November 2024"
            },
            {
                "date": "2024-11-11",
                "event": "Chrome distrust begins",
                "type": "distrust",
                "description": "Chrome 131 stops trusting new Entrust-issued TLS certificates"
            },
            {
                "date": "2024-11-15",
                "event": "Apple distrust begins",
                "type": "distrust",
                "description": "Safari/iOS stops trusting new Entrust TLS, S/MIME, and timestamping certificates"
            },
            {
                "date": "2024-11-30",
                "event": "Mozilla distrust begins",
                "type": "distrust",
                "description": "Firefox stops trusting new Entrust-issued TLS certificates"
            },
            {
                "date": "2025-01-29",
                "event": "Acquisition announced",
                "type": "acquisition",
                "description": "Sectigo announces purchase of Entrust's public certificate business"
            },
            {
                "date": "2025-04-16",
                "event": "Microsoft distrust begins",
                "type": "distrust",
                "description": "Windows/Edge stops trusting new Entrust-issued TLS certificates"
            },
            {
                "date": "2025-09-08",
                "event": "ECS portal end-of-life",
                "type": "migration",
                "description": "Entrust Certificate Services (ECS) portal enters read-only mode. No new certificate issuance."
            },
            {
                "date": "2025-09-25",
                "event": "Migration completed",
                "type": "milestone",
                "description": "Sectigo announces successful completion of largest CA migration in industry history. 500,000+ certificates transitioned."
            }
        ],
        "customerGuidance": {
            "whatHappened": "Sectigo acquired all of Entrust's public certificate business including TLS/SSL, S/MIME, code signing, document signing, eIDAS, and Verified Mark Certificates.",
            "whatNotAffected": "Entrust's private CA services, PKIaaS, and managed PKI products are NOT part of this acquisition.",
            "actionRequired": [
                "All ECS customers have been migrated to Sectigo Certificate Manager (SCM)",
                "Existing Entrust certificates remain valid until expiration",
                "New certificates are now issued by Sectigo",
                "Support now provided by Sectigo teams"
            ],
            "existingCertificates": "Certificates issued BEFORE browser distrust dates remain valid. Do NOT renew, rekey, or modify them through Entrust - this triggers distrust."
        },
        "sources": [
            {"name": "Sectigo Acquisition Announcement", "url": "https://www.sectigo.com/resource-library/sectigo-acquires-entrust-public-certificate-business"},
            {"name": "Entrust Announcement", "url": "https://www.entrust.com/company/newsroom/entrust-sells-public-certificate-business-to-sectigo"},
            {"name": "Migration Portal", "url": "https://www.sectigo.com/united-in-trust"},
            {"name": "Migration Completion", "url": "https://www.sectigo.com/resource-library/sectigo-completes-entrust-migration"}
        ],
        "relatedDeadlines": ["chrome-entrust-distrust", "apple-entrust-distrust", "mozilla-entrust-distrust", "microsoft-entrust-distrust"]
    }
]

# =============================================================================
# CA CERTIFICATE CHAINS
# Authoritative chain information for major CAs
# Last verified: 2025-12-29
# 
# MONTHLY UPDATE CHECKLIST:
# - [ ] Check Sectigo knowledge base for new intermediates
# - [ ] Check DigiCert root certificates page
# - [ ] Check Let's Encrypt /certificates page for hierarchy changes
# - [ ] Check GlobalSign support for new roots/intermediates
# - [ ] Check GoDaddy repository for updates
# - [ ] Verify all download URLs are still active
# =============================================================================

CA_CHAINS = [
    {
        "id": "sectigo",
        "name": "Sectigo (formerly Comodo)",
        "website": "https://www.sectigo.com",
        "chainBundleUrl": "https://support.sectigo.com/articles/Knowledge/Sectigo-Intermediate-Certificates",
        "lastVerified": "2025-12-29",
        "notes": "Sectigo acquired Entrust's public certificate business in January 2025. Transitioned to new R46/E46 hierarchy in 2025.",
        "roots": [
            {
                "name": "USERTrust RSA Certification Authority",
                "subject": "C=US, ST=New Jersey, L=Jersey City, O=The USERTRUST Network, CN=USERTrust RSA Certification Authority",
                "fingerprint_sha256": "E793C9B02FD8AA13E21C31228ACCB08119643B749C898964B1746D46C3D4CBD2",
                "fingerprint_sha1": "2B8F1B57330DBBA2D07A6C51F70EE90DDAB9AD8E",
                "validUntil": "2038-01-18",
                "keyType": "RSA 4096",
                "crtShId": "1199354",
                "downloadUrl": "https://crt.sh/?d=1199354",
                "status": "active"
            },
            {
                "name": "USERTrust ECC Certification Authority",
                "subject": "C=US, ST=New Jersey, L=Jersey City, O=The USERTRUST Network, CN=USERTrust ECC Certification Authority",
                "fingerprint_sha256": "4FF460D54B9C86DABFBCFC5712E0400D2BED3FBC4D4FBDAA86E06ADCD2A9AD7A",
                "fingerprint_sha1": "D1CBCA5DB2D52A7F693B674DE5F05A1D0C957DF0",
                "validUntil": "2038-01-18",
                "keyType": "ECC P-384",
                "crtShId": "2841410",
                "downloadUrl": "https://crt.sh/?d=2841410",
                "status": "active"
            },
            {
                "name": "COMODO RSA Certification Authority",
                "subject": "C=GB, ST=Greater Manchester, L=Salford, O=COMODO CA Limited, CN=COMODO RSA Certification Authority",
                "fingerprint_sha256": "52F0E1C4E58EC629291B60317F074671B85D7EA80D5B07273463534B32B40234",
                "validUntil": "2038-01-18",
                "keyType": "RSA 4096",
                "crtShId": "1720081",
                "downloadUrl": "https://crt.sh/?d=1720081",
                "status": "active",
                "notes": "Legacy root, still widely trusted"
            },
            {
                "name": "COMODO ECC Certification Authority",
                "subject": "C=GB, ST=Greater Manchester, L=Salford, O=COMODO CA Limited, CN=COMODO ECC Certification Authority",
                "fingerprint_sha256": "1793927A0614549789ADCE2F8F34F7F0B66D0F3AE3A3B84D21EC15DBBA4FADC7",
                "validUntil": "2038-01-18",
                "keyType": "ECC P-384",
                "crtShId": "2835394",
                "downloadUrl": "https://crt.sh/?d=2835394",
                "status": "active",
                "notes": "Legacy root, still widely trusted"
            }
        ],
        "intermediates": [
            {
                "name": "Sectigo Public Server Authentication Root R46",
                "use": "Current RSA TLS hierarchy (2025+)",
                "issuer": "USERTrust RSA Certification Authority",
                "validUntil": "2038-01-18",
                "downloadUrl": "https://sectigo.tbs-certificats.com/SectigoPublicServerAuthenticationRootR46_USERTrust.crt",
                "crlUrl": "http://crl.comodoca.com/SectigoPublicServerAuthenticationRootR46.crl",
                "status": "active"
            },
            {
                "name": "Sectigo Public Server Authentication Root E46",
                "use": "Current ECC TLS hierarchy (2025+)",
                "issuer": "USERTrust ECC Certification Authority",
                "validUntil": "2038-01-18",
                "downloadUrl": "https://sectigo.tbs-certificats.com/SectigoPublicServerAuthenticationRootE46_USERTrust.crt",
                "crlUrl": "http://crl.comodoca.com/SectigoPublicServerAuthenticationRootE46.crl",
                "status": "active"
            }
        ],
        "bundleDownloads": {
            "rsaIntermediates": "https://support.sectigo.com/Com_KnowledgeDetailPage?Id=kA01N000000rfBO",
            "eccIntermediates": "https://www.sectigo.com/knowledge-base/detail/Sectigo-Intermediate-Certificates-ECC/kA01N000000rfGE",
            "rootCertificates": "https://www.sectigo.com/knowledge-base/detail/Sectigo-Root-Certificates",
            "documentation": "https://support.sectigo.com/articles/Knowledge/Sectigo-Certificate-Installation-Guides"
        }
    },
    {
        "id": "digicert",
        "name": "DigiCert",
        "website": "https://www.digicert.com",
        "chainBundleUrl": "https://www.digicert.com/kb/digicert-root-certificates.htm",
        "lastVerified": "2025-12-29",
        "notes": "Acquired Symantec's certificate business in 2017. Major enterprise CA.",
        "roots": [
            {
                "name": "DigiCert Global Root G2",
                "subject": "C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root G2",
                "fingerprint_sha256": "CB3CCBB76031E5E0138F8DD39A23F9DE47FFC35E43C1144CEA27D46A5AB1CB5F",
                "validUntil": "2038-01-15",
                "keyType": "RSA 2048",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertGlobalRootG2.crt",
                "status": "active",
                "notes": "Primary root for new certificates"
            },
            {
                "name": "DigiCert Global Root CA",
                "subject": "C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root CA",
                "fingerprint_sha256": "4348A0E9444C78CB265E058D5E8944B4D84F9662BD26DB257F8934A443C70161",
                "validUntil": "2031-11-10",
                "keyType": "RSA 2048",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertGlobalRootCA.crt",
                "status": "active",
                "notes": "Legacy root, widely deployed"
            },
            {
                "name": "DigiCert Trusted Root G4",
                "subject": "C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Trusted Root G4",
                "fingerprint_sha256": "552F7BDCF1A7AF9E6CE672017F4F12ABF77240C78E761AC203D1D9D20AC89988",
                "validUntil": "2046-01-15",
                "keyType": "RSA 4096",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertTrustedRootG4.crt",
                "status": "active",
                "notes": "Newer G4 hierarchy"
            }
        ],
        "intermediates": [
            {
                "name": "DigiCert TLS RSA SHA256 2020 CA1",
                "use": "Standard OV/DV TLS certificates",
                "issuer": "DigiCert Global Root CA",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertTLSRSASHA2562020CA1-1.crt",
                "status": "active"
            },
            {
                "name": "DigiCert G5 TLS RSA4096 SHA384 2021 CA1",
                "use": "G5 TLS certificates",
                "issuer": "DigiCert Global Root G2",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertG5TLSRSA4096SHA3842021CA1.crt",
                "status": "active"
            },
            {
                "name": "DigiCert Global G2 TLS RSA SHA256 2020 CA1",
                "use": "G2 hierarchy TLS certificates",
                "issuer": "DigiCert Global Root G2",
                "downloadUrl": "https://cacerts.digicert.com/DigiCertGlobalG2TLSRSASHA2562020CA1.crt",
                "status": "active"
            }
        ],
        "bundleDownloads": {
            "allRootsAndIntermediates": "https://www.digicert.com/kb/digicert-root-certificates.htm",
            "trustedRoots": "https://knowledge.digicert.com/general-information/digicert-trusted-root-authority-certificates",
            "documentation": "https://docs.digicert.com/en/certcentral.html"
        }
    },
    {
        "id": "letsencrypt",
        "name": "Let's Encrypt (ISRG)",
        "website": "https://letsencrypt.org",
        "chainBundleUrl": "https://letsencrypt.org/certificates/",
        "lastVerified": "2025-12-29",
        "notes": "Free, automated CA. 90-day certificate validity. ACME protocol required. Active intermediates rotated to E7/E8/R12/R13 in late 2024.",
        "roots": [
            {
                "name": "ISRG Root X1",
                "subject": "O=Internet Security Research Group, CN=ISRG Root X1",
                "fingerprint_sha256": "96BCEC06264976F37460779ACF28C5A7CFE8A3C0AAE11A8FFCEE05C0BDDF08C6",
                "validUntil": "2030-06-04",
                "keyType": "RSA 4096",
                "downloadUrl": "https://letsencrypt.org/certs/isrgrootx1.pem",
                "derUrl": "https://letsencrypt.org/certs/isrgrootx1.der",
                "crtShId": "9314791",
                "status": "active",
                "notes": "Primary root, nearly ubiquitous trust"
            },
            {
                "name": "ISRG Root X2",
                "subject": "O=Internet Security Research Group, CN=ISRG Root X2",
                "fingerprint_sha256": "69729B8E15A86EFC177A57AFB7171DFC64ADD28C2FCA8CF1507E34453CCB1470",
                "validUntil": "2035-09-04",
                "keyType": "ECC P-384",
                "downloadUrl": "https://letsencrypt.org/certs/isrg-root-x2.pem",
                "derUrl": "https://letsencrypt.org/certs/isrg-root-x2.der",
                "crtShId": "3335562555",
                "status": "active",
                "notes": "ECDSA root, smaller chains when used directly"
            }
        ],
        "intermediates": [
            {
                "name": "Let's Encrypt E7",
                "use": "ECDSA leaf certificates (primary active)",
                "keyType": "ECDSA P-384",
                "issuer": "ISRG Root X2 (also cross-signed by X1)",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/e7.pem",
                "crossSignedUrl": "https://letsencrypt.org/certs/2024/e7-cross.pem",
                "crtShId": "12396132900",
                "status": "active"
            },
            {
                "name": "Let's Encrypt E8",
                "use": "ECDSA leaf certificates (secondary active)",
                "keyType": "ECDSA P-384",
                "issuer": "ISRG Root X2 (also cross-signed by X1)",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/e8.pem",
                "crossSignedUrl": "https://letsencrypt.org/certs/2024/e8-cross.pem",
                "crtShId": "12396132890",
                "status": "active"
            },
            {
                "name": "Let's Encrypt R12",
                "use": "RSA leaf certificates (primary active)",
                "keyType": "RSA 2048",
                "issuer": "ISRG Root X1",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/r12.pem",
                "crtShId": "12396132898",
                "status": "active"
            },
            {
                "name": "Let's Encrypt R13",
                "use": "RSA leaf certificates (secondary active)",
                "keyType": "RSA 2048",
                "issuer": "ISRG Root X1",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/r13.pem",
                "crtShId": "12396132902",
                "status": "active"
            },
            {
                "name": "Let's Encrypt R10",
                "use": "RSA leaf certificates (retired but valid)",
                "keyType": "RSA 2048",
                "issuer": "ISRG Root X1",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/r10.pem",
                "status": "retired",
                "notes": "No longer issuing, certificates still valid"
            },
            {
                "name": "Let's Encrypt R11",
                "use": "RSA leaf certificates (retired but valid)",
                "keyType": "RSA 2048",
                "issuer": "ISRG Root X1",
                "validUntil": "2027-03-12",
                "downloadUrl": "https://letsencrypt.org/certs/2024/r11.pem",
                "status": "retired",
                "notes": "No longer issuing, certificates still valid"
            }
        ],
        "upcomingIntermediates": [
            {
                "name": "Let's Encrypt YE1/YE2/YE3",
                "use": "Future ECDSA intermediates (expected 2026)",
                "keyType": "ECDSA P-384",
                "status": "upcoming",
                "notes": "New hierarchy expected mid-2026"
            },
            {
                "name": "Let's Encrypt YR1/YR2/YR3",
                "use": "Future RSA intermediates (expected 2026)",
                "keyType": "RSA 2048",
                "status": "upcoming",
                "notes": "New hierarchy expected mid-2026"
            }
        ],
        "bundleDownloads": {
            "chainOfTrust": "https://letsencrypt.org/certificates/",
            "documentation": "https://letsencrypt.org/docs/",
            "compatibility": "https://letsencrypt.org/docs/certificate-compatibility/"
        },
        "acmeEndpoints": {
            "production": "https://acme-v02.api.letsencrypt.org/directory",
            "staging": "https://acme-staging-v02.api.letsencrypt.org/directory"
        },
        "testUrls": {
            "validX1": "https://valid-isrgrootx1.letsencrypt.org/",
            "revokedX1": "https://revoked-isrgrootx1.letsencrypt.org/",
            "expiredX1": "https://expired-isrgrootx1.letsencrypt.org/",
            "validX2": "https://valid-isrgrootx2.letsencrypt.org/",
            "revokedX2": "https://revoked-isrgrootx2.letsencrypt.org/",
            "expiredX2": "https://expired-isrgrootx2.letsencrypt.org/"
        }
    },
    {
        "id": "globalsign",
        "name": "GlobalSign",
        "website": "https://www.globalsign.com",
        "chainBundleUrl": "https://support.globalsign.com/ca-certificates/root-certificates/globalsign-root-certificates",
        "lastVerified": "2025-12-29",
        "notes": "Major enterprise CA. Strong in Europe and Asia. AlphaSSL is their DV brand.",
        "roots": [
            {
                "name": "GlobalSign Root CA - R3",
                "subject": "OU=GlobalSign Root CA - R3, O=GlobalSign, CN=GlobalSign",
                "fingerprint_sha256": "CBB522D7B7F127AD6A0113865BDF1CD4102E7D0759AF635A7CF4720DC963C53B",
                "fingerprint_sha1": "D69B561148F01C77C54578C10926DF5B856976AD",
                "validUntil": "2029-03-18",
                "keyType": "RSA 2048",
                "downloadUrl": "http://secure.globalsign.com/cacert/root-r3.crt",
                "crlUrl": "http://crl.globalsign.net/root-r3.crl",
                "status": "active",
                "notes": "Most ubiquitous GlobalSign root"
            },
            {
                "name": "GlobalSign Root CA - R6",
                "subject": "OU=GlobalSign Root CA - R6, O=GlobalSign, CN=GlobalSign",
                "fingerprint_sha256": "2CABEAFE37D06CA22ABA7391C0033D25982952C453647349763A3AB5AD6CCF69",
                "validUntil": "2034-12-10",
                "keyType": "RSA 4096",
                "downloadUrl": "http://secure.globalsign.com/cacert/root-r6.crt",
                "status": "active",
                "notes": "Newer, more secure root"
            },
            {
                "name": "GlobalSign Root CA - R46",
                "subject": "CN=GlobalSign Root R46, O=GlobalSign nv-sa, C=BE",
                "validUntil": "2046+",
                "keyType": "RSA 4096",
                "status": "active",
                "notes": "Newest RSA root, cross-signed by R3 for compatibility"
            }
        ],
        "intermediates": [
            {
                "name": "GlobalSign GCC R6 AlphaSSL CA 2025",
                "use": "DV certificates (AlphaSSL brand)",
                "fingerprint_sha1": "431955E6E5DABE857F1336C02368E5495F143EED",
                "issuer": "GlobalSign Root CA - R6",
                "validUntil": "2027-05-21",
                "status": "active"
            }
        ],
        "bundleDownloads": {
            "allCertificates": "https://support.globalsign.com/ca-certificates/root-certificates/globalsign-root-certificates",
            "crossCertificates": "https://support.globalsign.com/ca-certificates/globalsign-cross-certificates",
            "documentation": "https://support.globalsign.com/"
        }
    },
    {
        "id": "godaddy",
        "name": "GoDaddy / Starfield",
        "website": "https://www.godaddy.com/web-security/ssl-certificate",
        "chainBundleUrl": "https://certs.godaddy.com/repository",
        "lastVerified": "2025-12-29",
        "notes": "Popular with small businesses and hosting customers. Starfield is their alternate brand. Generated new R1 root in August 2025.",
        "roots": [
            {
                "name": "Go Daddy Root Certificate Authority - G2",
                "subject": "C=US, ST=Arizona, L=Scottsdale, O=GoDaddy.com, Inc., CN=Go Daddy Root Certificate Authority - G2",
                "fingerprint_sha256": "45140B3247EB9CC8C5B4F0D7B53091F73292089E6E5A63E2749DD3ACA9198EDA",
                "fingerprint_sha1": "47BEABC922EAE80E78783462A79F45C254FDE68B",
                "validUntil": "2037-12-31",
                "keyType": "RSA 2048",
                "downloadUrl": "https://certs.godaddy.com/repository/gdroot-g2.crt",
                "crlUrl": "http://crl.godaddy.com/gdroot-g2.crl",
                "status": "active"
            },
            {
                "name": "Go Daddy Class 2 Certification Authority",
                "subject": "C=US, O=The Go Daddy Group, Inc., OU=Go Daddy Class 2 Certification Authority",
                "fingerprint_sha256": "C3846BF24B9E93CA64274C0EC67C1ECC5E024FFCACD2D74019350E81FE546AE4",
                "validUntil": "2034-06-29",
                "keyType": "RSA 2048",
                "downloadUrl": "https://certs.godaddy.com/repository/gd-class2-root.crt",
                "status": "legacy",
                "notes": "Legacy root, G2 cross-signed to this"
            }
        ],
        "intermediates": [
            {
                "name": "Go Daddy Secure Certificate Authority - G2",
                "use": "Standard SSL certificates",
                "fingerprint_sha256": "973A41276FFD01E027A2AAD49E34C37846D3E976FF6A620B6712E33832041AA6",
                "fingerprint_sha1": "27AC9369FAF25207BB2627CEFACCBE4EF9C319B8",
                "issuer": "Go Daddy Root Certificate Authority - G2",
                "validUntil": "2031-05-03",
                "downloadUrl": "https://certs.godaddy.com/repository/gdig2.crt",
                "status": "active"
            }
        ],
        "bundleDownloads": {
            "repository": "https://certs.godaddy.com/repository",
            "documentation": "https://www.godaddy.com/help/ssl-certificate-installation-and-use-32124"
        }
    }
]

# =============================================================================
# CA CHAIN QUICK REFERENCE
# Common troubleshooting scenarios and verification commands
# =============================================================================

CA_CHAIN_QUICK_REFERENCE = {
    "description": "Quick reference for common chain-related issues",
    "commonProblems": [
        {
            "symptom": "Certificate not trusted on older devices/browsers",
            "likelyCause": "Missing intermediate certificate in server configuration",
            "fix": "Ensure full chain is configured on server, not just leaf certificate. Download intermediate from CA's website.",
            "severity": "high"
        },
        {
            "symptom": "Chain contains cross-signed certificate warnings",
            "likelyCause": "Using legacy cross-sign instead of direct chain",
            "fix": "Update to current intermediate from CA's website",
            "severity": "low"
        },
        {
            "symptom": "openssl verify shows 'unable to get local issuer certificate'",
            "likelyCause": "Root CA not in local trust store or chain incomplete",
            "fix": "Download root from CA or check intermediate configuration",
            "severity": "medium"
        },
        {
            "symptom": "Different behavior between browsers",
            "likelyCause": "Some browsers have intermediate caching (AIA fetching), others don't",
            "fix": "Always send full chain from server - don't rely on AIA",
            "severity": "high"
        },
        {
            "symptom": "ERR_CERT_AUTHORITY_INVALID in Chrome",
            "likelyCause": "Intermediate or root not trusted, or chain incomplete",
            "fix": "Check chain order (leaf -> intermediate -> root) and verify all certs are from same CA",
            "severity": "high"
        },
        {
            "symptom": "Certificate works in browser but fails in API/curl",
            "likelyCause": "Server OS/container trust store outdated or missing CA bundle",
            "fix": "Update ca-certificates package or explicitly provide CA bundle",
            "severity": "medium"
        }
    ],
    "verificationCommands": {
        "checkServerChain": "openssl s_client -connect example.com:443 -showcerts",
        "verifyChainFile": "openssl verify -CAfile chain.pem certificate.pem",
        "viewCertificate": "openssl x509 -in cert.pem -text -noout",
        "checkFingerprint": "openssl x509 -in cert.pem -fingerprint -sha256 -noout",
        "checkExpiration": "openssl x509 -in cert.pem -noout -enddate",
        "checkIssuer": "openssl x509 -in cert.pem -noout -issuer",
        "downloadFromServer": "openssl s_client -connect example.com:443 -servername example.com 2>/dev/null | openssl x509 -outform PEM > server.pem"
    },
    "usefulTools": [
        {
            "name": "SSL Labs Server Test",
            "url": "https://www.ssllabs.com/ssltest/",
            "use": "Comprehensive SSL/TLS testing including chain validation"
        },
        {
            "name": "crt.sh",
            "url": "https://crt.sh/",
            "use": "Certificate Transparency log search - find any certificate by domain or fingerprint"
        },
        {
            "name": "What's My Chain Cert?",
            "url": "https://whatsmychaincert.com/",
            "use": "Automatically generate correct intermediate chain bundle"
        },
        {
            "name": "Certificate Search",
            "url": "https://search.censys.io/",
            "use": "Search certificates by various attributes"
        },
        {
            "name": "SSL Checker",
            "url": "https://www.sslshopper.com/ssl-checker.html",
            "use": "Quick chain and expiration check"
        }
    ],
    "chainOrderReminder": {
        "correct": ["Leaf/End-entity certificate", "Intermediate CA(s)", "Root CA (optional - usually in trust store)"],
        "notes": "Most servers should NOT include the root CA - it should be in the client's trust store. Including it adds unnecessary bytes to each TLS handshake."
    }
}

# Related RFCs
RELATED_RFCS = [
    {"rfc": "RFC 5280", "title": "X.509 PKI Certificate Profile", "url": "https://datatracker.ietf.org/doc/html/rfc5280"},
    {"rfc": "RFC 6960", "title": "OCSP", "url": "https://datatracker.ietf.org/doc/html/rfc6960"},
    {"rfc": "RFC 6962", "title": "Certificate Transparency", "url": "https://datatracker.ietf.org/doc/html/rfc6962"},
    {"rfc": "RFC 8659", "title": "CAA Records", "url": "https://datatracker.ietf.org/doc/html/rfc8659"},
    {"rfc": "NIST FIPS 203", "title": "ML-KEM (Key Encapsulation)", "url": "https://csrc.nist.gov/pubs/fips/203/final"},
    {"rfc": "NIST FIPS 204", "title": "ML-DSA (Digital Signatures)", "url": "https://csrc.nist.gov/pubs/fips/204/final"},
    {"rfc": "NIST FIPS 205", "title": "SLH-DSA (Hash-Based Signatures)", "url": "https://csrc.nist.gov/pubs/fips/205/final"},
]

# Metadata for the compliance hub
COMPLIANCE_METADATA = {
    "lastUpdated": "2026-08-15",
    "dataVersion": "2.4.19",
    "basedOn": "CA/B Forum TLS BR 2.2.9, Code Signing BR 3.11, EV Guidelines 2.0.3, S/MIME BR 1.0.15, SC-080/081/085/090/091/092/097/098/099/101 Ballots, SMC017v2, Chrome Root Program v1.8, Mozilla Root Store Policy v3.1, Apple Root Program Policy v2.0, Microsoft Trusted Root Program Requirements v1.2, NIST SP 800-131A Rev 3 (initial public draft), NIST SP 800-57 Rev 5, NIST FIPS 203/204/205 (PQC), NIST IR 8547 (initial public draft), NSA CNSA 2.0, PCI DSS v4.0.1, DORA (EU), NIS2 (EU), UK CSR Bill",
    "disclaimer": "This is a community resource for educational purposes. Always verify against official sources before making compliance decisions.",
    "sources": [
        "https://cabforum.org",
        "https://g.co/chrome/root-policy",
        "https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/policy/",
        "https://github.com/apple/apple-root-program",
        "https://github.com/TrustedRootProgram/Program-Requirements",
        "https://aka.ms/RootCert",
        "https://letsencrypt.org/certificates/",
        "https://www.sectigo.com/knowledge-base/detail/Sectigo-Root-Certificates",
        "https://www.digicert.com/kb/digicert-root-certificates.htm",
        "https://support.globalsign.com/ca-certificates/root-certificates",
        "https://certs.godaddy.com/repository"
    ]
}

DATA_FRESHNESS = {
    # Bumped 2026-08-15 on Pat's instruction, and this one is a DEPARTURE from
    # the standing targeted-apply convention (which bumps lastUpdated +
    # dataVersion + fieldVerifications only, precisely so a small apply cannot
    # push the staleness banner out). Recorded so nobody reads it as routine:
    # the banner moves 2026-09-14 -> 2026-09-29. What backs it is the day's
    # actual scope rather than the three-entry apply alone — Apple Root Program
    # Policy v2.0 read end to end against the primary source, and a back-check
    # of all fifteen April-to-August "no date certain" classifications plus the
    # unverified live drafts, which returned eleven correctly filed against
    # primary sources and two real in-force obligations (Czechia, Sweden, both
    # already tracked).
    "lastFullReview": "2026-08-15",
    "nextReviewDue": "2026-09-14",
    "reviewIntervalDays": 30,
    "fieldVerifications": {
        "deadlines": {"verified": "2026-08-15", "source": "2026-08-15 targeted CORRECTION to microsoft-august-2026-root-disable (no new entry, no date change, lastFullReview untouched): the entry OVERSTATED the blast radius of the Baltimore CyberTrust Root disable and buried the root that actually matters. Baltimore CyberTrust Root's validity is 2000-05-12 to 2025-05-12 23:59:00 GMT — it EXPIRED fifteen months before this release, so anything still chaining to it stopped validating in May 2025 and Microsoft disabling it on 2026-08-25 is bookkeeping on a dead root. The prior text called it 'a long-lived, heavily embedded root' whose exposure sits in appliances, IoT, pinned trust stores and separately-maintained Java/OpenSSL stores, which reads as a live root about to break; that clause is REMOVED and replaced with the expiry fact plus the residual hygiene work it actually implies (stale bundled stores and vendor images, monitoring/pinning on the thumbprint, cacerts copied off old hosts). GeoTrust Universal CA is PROMOTED to first position in title, description, impact and scenario: its validity is 2004-03-04 to 2029-03-04, so it is the only root in this release whose disable breaks certificates that work today. What did NOT change and was re-affirmed as correct: the date 2026-08-25 and its NOTE (the notice states only the NotBefore date and lists Disable/Remove without restating a date), both SHA-1 thumbprints, the Visa Information Delivery Root CA removal, isMajor True, and the generic disable-vs-NotBefore mechanism statement — 'a disable breaks every certificate under the root regardless of issuance date' is accurate as mechanism and is kept; it is simply not consequential for an already-expired root, which the enforcement text now says in terms. Source of record unchanged and re-cited: TrustedRootProgram/Program-Requirements, trusted-root/2026/august-2026.md at commit 13bfd4f (2026-08-05), 'This release will Disable' listing DigiCert \\ Baltimore CyberTrust Root \\ D4DE20D05E66FC53FE1A50882C78DB2852CAE474 and DigiCert \\ GeoTrust Universal CA \\ E621F3354379059A4B68309D8A2F74221587EC79. pki_check_all_documents was deliberately NOT run: the feed had already flagged this entry and the correction is a reading of root validity periods, not a document-state question. ALSO: microsoft-august-2026-root-disable gained an EXPLICIT_RELATED_GUIDES mapping to the Root Stores guide, matching the four Entrust distrust cards ('certificates' has no category default). The dedicated /guides/digicert-global-root-g2 guide built on these same facts is queued for the 2026-08-18 batch and is deliberately NOT linked yet — it is not in the content tracker, so a chip would render a 404 for three days; append it there once ct_get_content resolves it. Prior review 2026-08-15 targeted apply (NOT a full review; lastFullReview stays 2026-07-31): Apple Root Program Policy v2.0 applied, from the primary instrument fetched directly (https://raw.githubusercontent.com/apple/apple-root-program/main/policy.md, HTTP 200, 31247 bytes) rather than from a report. The document's header reads 'Version 2.0 / Effective 2026-08-01' and its Change Log row for v2.0 dates every obligation below; each was also read in its normative section. THREE entries, split by date because the dates differ: apple-policy-v2-subca-eku (2026-08-01, ongoing), apple-policy-v2-smime-rfc822name (2027-02-01), apple-policy-v2-single-trust-purpose (2027-07-01). (1) 2026-08-01, §1.7: a Subordinate CA Certificate signed on or after that date MUST contain an EKU extension and MUST NOT assert anyExtendedKeyUsage (2.5.29.37.0); signed on or after 2026-08-01 and before 2027-07-01 it must also be single-Trust-Purpose OR carry no Appendix A Trust-Purpose EKU under a multi-purpose root with an audited CP/CPS covering the use case. Bundled into the same entry because they share the date: §1.4's Apple approval before issuing a Sub-CA (or cross-sign) to an Externally Operated Subordinate CA, and §1.5's RSA 4096-bit / ECDSA 384-bit minimum for Root Inclusion Requests. CLASSIFIED ONGOING, deliberately, and this is the reusable part: §1.7 states that a renewal, re-key or cross-sign produces a NEW Subordinate CA Certificate whose signing date is the date of that issuance, so the rule bites again on every future Sub-CA signing rather than having transitioned once — type (b), added to ONGOING_IDS. CAVEAT recorded honestly: §1.4 and §1.5 state their requirements WITHOUT restating the date; only the Change Log attaches 2026-08-01 to them, which the entry text says in terms. (2) 2027-02-01, §2.3: all newly signed Subscriber certificates containing id-kp-emailProtection MUST include at least one rfc822Name in subjectAltName; upcoming, not ongoing. (3) 2027-07-01: §1.7 single Trust Purpose mandatory (the carve-out ends), §1.3.1 all Policy Documents must be Markdown CP/CPS scoped to a single Trust Purpose (only Server Authentication + Legacy TLS, or Secure Email + Legacy S/MIME, may combine), and §1.2.3.1 Detailed Controls Report attestation duties for audit periods starting on or after that date. Mozilla's own DCR requirement (mozilla-dcr-audit-periods, MRSP v3.1) is the SAME date and is deliberately NOT deduped against it — separate root programs, separate obligations on the same CA. Per the §2 note, issuance-related effective dates run from 00:00:00 UTC. Provenance: these reached us as COWORK's 2026-08-15T12:42:00Z handover after the 08-12 pipeline misclassified the policy as a content candidate under 'no-date-certain'; every date here was re-read from the source, not taken from the report. Czechia and Sweden from the same morning's back-check were checked and NOT added — nis2-czechia-effective (2025-11-01) and nis2-sweden-effective (2026-01-15) already exist in the NIS2 framework sub-list, both already ongoing and in ONGOING_IDS since 2026-07-30. nis2-netherlands-cbw promoted to status ongoing on its commencement date (2026-08-15) per the plan recorded in the 2026-07-31 note below; it could not carry the key while future-dated. Prior review 2026-08-07 targeted apply (NOT a full review; lastFullReview stays 2026-07-31): Microsoft August 2026 Trusted Root Program deployment notice applied, from the primary instrument fetched directly (https://raw.githubusercontent.com/TrustedRootProgram/Program-Requirements/main/trusted-root/2026/august-2026.md, HTTP 200, 9336 bytes) rather than from a report. Two date-certain obligations, both day-precise IN THE SOURCE with no derivation: release 2026-08-25 ('On Tuesday, August 25, 2026, Microsoft released an update...', matching the document's own ms.date front matter) and NotBefore 2026-09-15 ('The NotBefore date is set to September 15, 2026. This means only certificates issued after this date will be distrusted'). Split into TWO entries deliberately, departing from the single-entry April 2026 precedent, because the release carries two DIFFERENT mechanisms on two different dates: microsoft-august-2026-root-disable (2026-08-25) for the outright Disable of Baltimore CyberTrust Root and GeoTrust Universal CA plus the Removal of Visa Information Delivery Root CA, which breaks every certificate under those roots regardless of issuance date; and microsoft-august-2026-ctl-notbefore (2026-09-15) for the NotBefore sets, which break only new issuance. Folding a root disable into a 'renewals break' entry would have understated it. Counts reconciled against the source: 8 ML-DSA PQC pilot roots added, 19 roots fully NotBefore'd (9 Entrust/AffirmTrust + 6 SecureTrust/Trustwave + 4 others), 2 disabled, 1 removed, per-EKU NotBefores of 3 code signing / 10 S/MIME / 4 time stamping / 3 server auth / 2 client auth / 1 document signing. Neither entry persists status - both are future-dated and compute 'upcoming'. CAVEAT recorded honestly: the notice states only the one NotBefore date and lists Disable/Remove as separate actions of the release without restating a date for them, so 2026-08-25 for the disable entry is the release date, not a date the source attaches to the disable action. This corroborates COWORK's 2026-08-05T18:36:00Z FINDING in every particular checked; it is now independently established rather than an attributed claim. SIDE EFFECT worth knowing: 2026-09-15 now carries FOUR obligations, not the three recorded in the 2026-07-31 note below - SC-097 SHA-1 CA/CRL sunset, CSC-32 reserved policy OID, the S/MIME CA RSA-4096 key floor, and now this Microsoft NotBefore. Prior review 2026-08-06 targeted apply: SC0101v2 HOLD RELEASED and applied. Its IPR Review Period (2026-07-07 08:00 to 2026-08-06 08:00 UTC, verbatim from the ballot page's Notice of Review Period) closed with NO Notice to Exclude Essential Claims. Verified in the live venue - https://groups.google.com/a/groups.cabforum.org/g/public, readable anonymously - because lists.cabforum.org/pipermail is a DEAD host, not a frozen archive. The whole list shows no activity after 2026-07-30; plain-term searches (Google Groups group-search silently zeroes any query carrying after:/before:, so date operators must never be used) return newest 'exclusion' 2026-04-29, newest 'essential claims' 2026-02-19, and one 'SC101' hit in the 2026-07-02 plenary minutes. Publication confirmed against the normative text, not the ballot page alone: cabforum/servercert main merged 'SC-101v2: Clarify Authorization Domain Names (#627)' at 2026-08-06T12:30Z and docs/BR.md now carries version-history row 2.2.9 | SC101 | adopted 2026-07-02 | effective 2026-08-06, the 2026-11-15 row in the Section 1.2.2 Relevant Dates table, and the Section 3.2.2.4 transition sentence. Added sc101v2-adn-derivation-mandatory (2026-11-15, major, upcoming). NOTE the fallback names BR Version 2.2.7, not 2.2.8 - a CA may comply with 3.2.2.4 of v2.2.7 until 2026-11-15. TLS BR doc version BUMPED 2.2.8 -> 2.2.9 (CABF_DOCUMENTS tls-br, date Jun -> Aug 2026) and basedOn updated, on Pat's instruction the same day, reversing the initial decision to leave it queued behind the tlsbr hold. Justified by the normative text: BR.md's own version-history table stamps 2.2.9 effective 2026-08-06, which is the publication event. Two lagging surfaces do NOT contradict that and are expected to catch up - the BRs/v2.2.9 release tag was not yet cut and cabforum.org's documents page still listed 2.2.8 as Current Version when this was written. The tlsbr manual hold (to 2026-08-20) is deliberately NOT retired by this bump: it still queues AUTO-proposed doc bumps, which matters because SC102's IPR window does not close until 2026-08-13 and may land a further version; its expiry stays owned by the 2026-08-16 task. Prior review 2026-07-31: SMC017v2 HOLD RELEASED and applied. The IPR Review Period (2026-06-30 20:00 to 2026-07-30 20:00 UTC, per the ballot's own Review Notice) closed with no Exclusion Notice, confirmed by publication of S/MIME BR v1.0.15 on 2026-07-30 (cabforum/smime release tag Ballot_SMC017; the CABF S/MIME documents page lists v1.0.15 as adopted by SMC017v2). Added smc017-smime-ca-rsa-4096 (2026-09-15 - the trigger is key CREATION date, NOT certificate issuance date) and smc017-smime-subca-3072-issuance-sunset (2027-09-15); both upcoming, neither ongoing. smime-br doc bumped 1.0.14 -> 1.0.15. Netherlands NIS2 RESOLVED after three appearances: primary source is the Cyberbeveiligingsbesluit, Besluit van 8 juli 2026, Staatsblad 2026 nr. 189 (published 2026-07-10), which sets Cyberbeveiligingswet commencement at 2026-08-15; added nis2-netherlands-cbw at day precision, is_estimated false. It is an ONGOING_IDS candidate (type b, in-force regime) only AFTER 2026-08-15 - a future-dated ongoing entry breaks the date-consistency test. EV Guidelines discrepancy CLOSED: v2.0.3 dated 6 July 2026 adopted via SC087, so the auto-applied doc bump was correct; basedOn corrected (it still said 2.0.2). SC102 hold confirmed against its 2026-07-14 Review Notice (window 2026-07-14 08:00 to 2026-08-13 08:00 UTC), pin kept at 2026-08-15; SC0101v2 hold unchanged at 2026-08-08. 2026-09-15 now carries THREE obligations: SC-097 SHA-1 CA/CRL sunset, CSC-32 reserved policy OID, and the S/MIME CA RSA-4096 floor. Prior review 2026-07-29: uk-csr-lords-stage and uk-csr-royal-assent advanced from 'second reading scheduled' to 'completed 14 Jul 2026, now at Committee stage' (legislative stage only, no new date-certain; Royal Assent estimate unchanged at 2026-12-31); SMC017v2 held pending IPR close. STANDING RULE from that review, still true: a CA/B Forum ballot's IPR Review Period runs 30 days from the Review Notice, NOT from the vote-completion date - read the window off the ballot page, never re-derive it as vote+30d. Prior review 2026-07-21 (Microsoft TRP blind-window)"},
        "rootStores": {"verified": "2026-07-21", "source": "Individual root program policies. 2026-07-21: Microsoft source repointed to github.com/TrustedRootProgram/Program-Requirements (official since Oct 2025; superseded learn.microsoft.com page was monitored dead ~9 months) and converted from manual check to commits.atom feed. Blind-window review same day: Microsoft framework entry updated to Requirements v1.2 (single-purpose roots, 10-yr validity, incident reporting, CTLM) with April/June 2026 CTL actions."},
        "algorithmRequirements": {"verified": "2026-05-14", "source": "CA/B Forum TLS BR 2.2.6, NIST FIPS 203/204/205, NIST SP 800-131A Rev 3"},
        "caChains": {"verified": "2026-05-14", "source": "Official CA documentation"},
        "pqcStandards": {"verified": "2026-05-14", "source": "NIST FIPS 203/204/205, NSA CNSA 2.0"}
    },
    "staleAfterDays": 45
}

# =============================================================================
# NIST SP 800-131A Rev 2 - Transitioning the Use of Cryptographic Algorithms
# Source: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar2.pdf
# =============================================================================
NIST_800_131A = {
    "document": "NIST SP 800-131A",
    "version": "Revision 2",
    "effectiveDate": "2019-03-21",
    "sourceUrl": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar2.pdf",
    "lastVerified": "2024-12-15",
    "algorithms": [
        {
            "name": "RSA-1024",
            "type": "asymmetric",
            "status": "disallowed",
            "statusColor": "red",
            "disallowedDate": "2014-01-01",
            "notes": "Not acceptable for any cryptographic use"
        },
        {
            "name": "RSA-2048",
            "type": "asymmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": "2030-12-31",
            "notes": "Minimum key size for digital signatures"
        },
        {
            "name": "RSA-3072+",
            "type": "asymmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "Recommended for use beyond 2030"
        },
        {
            "name": "ECDSA P-256",
            "type": "asymmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "128-bit security strength"
        },
        {
            "name": "ECDSA P-384",
            "type": "asymmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "192-bit security strength"
        },
        {
            "name": "ECDSA P-521",
            "type": "asymmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "256-bit security strength"
        },
        {
            "name": "DSA-1024",
            "type": "asymmetric",
            "status": "disallowed",
            "statusColor": "red",
            "disallowedDate": "2014-01-01",
            "notes": "Not acceptable for any use"
        },
        {
            "name": "DSA-2048",
            "type": "asymmetric",
            "status": "deprecated",
            "statusColor": "yellow",
            "notes": "Legacy use only, not recommended"
        },
        {
            "name": "SHA-1 (signatures)",
            "type": "hash",
            "status": "disallowed",
            "statusColor": "red",
            "disallowedDate": "2014-01-01",
            "notes": "Not for digital signatures"
        },
        {
            "name": "SHA-1 (other uses)",
            "type": "hash",
            "status": "deprecated",
            "statusColor": "yellow",
            "notes": "Legacy applications only, not for security"
        },
        {
            "name": "SHA-224",
            "type": "hash",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": "2030-12-31",
            "notes": "112-bit security strength"
        },
        {
            "name": "SHA-256",
            "type": "hash",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "Required for most applications"
        },
        {
            "name": "SHA-384",
            "type": "hash",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "192-bit security strength"
        },
        {
            "name": "SHA-512",
            "type": "hash",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "256-bit security strength"
        },
        {
            "name": "MD5",
            "type": "hash",
            "status": "disallowed",
            "statusColor": "red",
            "disallowedDate": "2010-01-01",
            "notes": "Never use for any security purpose"
        },
        {
            "name": "3DES (Triple DES)",
            "type": "symmetric",
            "status": "deprecated",
            "statusColor": "yellow",
            "disallowedDate": "2023-12-31",
            "notes": "Disallowed after 2023 for new applications"
        },
        {
            "name": "AES-128",
            "type": "symmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "Approved for all uses"
        },
        {
            "name": "AES-192",
            "type": "symmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "Approved for all uses"
        },
        {
            "name": "AES-256",
            "type": "symmetric",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "notes": "Approved for all uses, highest security"
        },
        # Post-Quantum Cryptography (FIPS 203, 204, 205)
        {
            "name": "ML-KEM-512",
            "type": "pqc-kem",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 203",
            "securityLevel": 1,
            "notes": "NIST Security Level 1 (~AES-128 equivalent). Use for general key encapsulation."
        },
        {
            "name": "ML-KEM-768",
            "type": "pqc-kem",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 203",
            "securityLevel": 3,
            "notes": "NIST Security Level 3 (~AES-192 equivalent). Recommended for most applications."
        },
        {
            "name": "ML-KEM-1024",
            "type": "pqc-kem",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 203",
            "securityLevel": 5,
            "notes": "NIST Security Level 5 (~AES-256 equivalent). Highest security, larger keys."
        },
        {
            "name": "ML-DSA-44",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 204",
            "securityLevel": 2,
            "notes": "NIST Security Level 2. Smallest signatures of ML-DSA variants."
        },
        {
            "name": "ML-DSA-65",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 204",
            "securityLevel": 3,
            "notes": "NIST Security Level 3. Recommended for most digital signature applications."
        },
        {
            "name": "ML-DSA-87",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 204",
            "securityLevel": 5,
            "notes": "NIST Security Level 5. Highest security ML-DSA variant."
        },
        {
            "name": "SLH-DSA-128s",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 205",
            "securityLevel": 1,
            "notes": "Hash-based signature. Small signatures, slower signing. Good for firmware/code signing."
        },
        {
            "name": "SLH-DSA-128f",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 205",
            "securityLevel": 1,
            "notes": "Hash-based signature. Fast signing, larger signatures."
        },
        {
            "name": "SLH-DSA-192s",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 205",
            "securityLevel": 3,
            "notes": "Hash-based signature. Security Level 3, small signatures."
        },
        {
            "name": "SLH-DSA-256s",
            "type": "pqc-signature",
            "status": "acceptable",
            "statusColor": "green",
            "acceptableThrough": None,
            "fipsStandard": "FIPS 205",
            "securityLevel": 5,
            "notes": "Hash-based signature. Highest security SLH-DSA variant."
        }
    ],
    "keyStrengthTable": [
        {
            "securityBits": 80,
            "rsaKeySize": 1024,
            "eccKeySize": "160-223",
            "hashFunction": "SHA-1",
            "status": "disallowed",
            "notes": "Not acceptable for any federal use"
        },
        {
            "securityBits": 112,
            "rsaKeySize": 2048,
            "eccKeySize": "224-255",
            "hashFunction": "SHA-224/256/384/512",
            "status": "acceptable",
            "acceptableThrough": "2030-12-31",
            "notes": "Minimum for current use"
        },
        {
            "securityBits": 128,
            "rsaKeySize": 3072,
            "eccKeySize": "256-383",
            "hashFunction": "SHA-256/384/512",
            "status": "acceptable",
            "acceptableThrough": None,
            "notes": "Recommended for use beyond 2030"
        },
        {
            "securityBits": 192,
            "rsaKeySize": 7680,
            "eccKeySize": "384-511",
            "hashFunction": "SHA-384/512",
            "status": "acceptable",
            "acceptableThrough": None,
            "notes": "High security applications"
        },
        {
            "securityBits": 256,
            "rsaKeySize": 15360,
            "eccKeySize": "512+",
            "hashFunction": "SHA-512",
            "status": "acceptable",
            "acceptableThrough": None,
            "notes": "Highest security level"
        }
    ]
}

# =============================================================================
# PCI DSS v4.0.1 - Payment Card Industry Data Security Standard
# Source: https://docs-prv.pcisecuritystandards.org/PCI%20DSS/Standard/PCI-DSS-v4_0_1.pdf
# =============================================================================
PCI_DSS_V4 = {
    "document": "PCI DSS",
    "version": "4.0.1",
    "effectiveDate": "2024-03-31",
    "sourceUrl": "https://docs-prv.pcisecuritystandards.org/PCI%20DSS/Standard/PCI-DSS-v4_0_1.pdf",
    "lastVerified": "2024-12-15",
    "requirements": [
        {
            "id": "4.2.1",
            "title": "Strong cryptography for transmission",
            "description": "Strong cryptography and security protocols are used to safeguard PAN during transmission over open, public networks",
            "certificateRelevance": "high",
            "details": [
                "TLS 1.2 or higher required",
                "Only trusted certificates accepted",
                "Proper certificate validation required",
                "Industry-accepted cipher suites only"
            ]
        },
        {
            "id": "4.2.1.1",
            "title": "Certificate inventory",
            "description": "An inventory of trusted keys and certificates is maintained",
            "certificateRelevance": "high",
            "details": [
                "Inventory of all certificates",
                "Track expiration dates",
                "Document certificate purposes",
                "Monitor for unauthorized certificates"
            ]
        },
        {
            "id": "3.6.1",
            "title": "Cryptographic key management",
            "description": "Procedures are defined and implemented for cryptographic key management",
            "certificateRelevance": "high",
            "details": [
                "Strong key generation",
                "Secure key distribution",
                "Secure key storage",
                "Key rotation policies"
            ]
        },
        {
            "id": "3.7.1",
            "title": "Key management policies",
            "description": "Key-management policies and procedures include generation of strong cryptographic keys",
            "certificateRelevance": "medium",
            "details": [
                "RSA 2048-bit minimum",
                "ECC 224-bit minimum",
                "AES 128-bit minimum for symmetric"
            ]
        },
        {
            "id": "12.3.3",
            "title": "Cryptographic cipher suites and protocols",
            "description": "Cryptographic cipher suites and protocols in use are documented and reviewed",
            "certificateRelevance": "medium",
            "details": [
                "Document all protocols in use",
                "Review annually",
                "Remove deprecated algorithms",
                "Update when vulnerabilities discovered"
            ]
        }
    ],
    "deadlines": [
        {
            "date": "2024-03-31",
            "event": "PCI DSS v3.2.1 retired",
            "impact": "All assessments must use v4.0",
            "isMajor": True
        },
        {
            "date": "2025-03-31",
            "event": "Future-dated requirements mandatory",
            "impact": "All new v4.0 requirements become required",
            "isMajor": True
        }
    ],
    "tlsRequirements": {
        "minimumVersion": "TLS 1.2",
        "recommendedVersion": "TLS 1.3",
        "prohibitedVersions": ["SSL 2.0", "SSL 3.0", "TLS 1.0", "TLS 1.1"],
        "notes": "TLS 1.0 and 1.1 deprecated since June 30, 2018"
    }
}

# =============================================================================
# FIPS 140-2 / FIPS 140-3 - Security Requirements for Cryptographic Modules
# Source: https://csrc.nist.gov/publications/detail/fips/140/3/final
# =============================================================================
FIPS_140 = {
    "currentVersion": "FIPS 140-3",
    "previousVersion": "FIPS 140-2",
    "sourceUrl": "https://csrc.nist.gov/publications/detail/fips/140/3/final",
    "lastVerified": "2024-12-15",
    "securityLevels": [
        {
            "level": 1,
            "name": "Level 1",
            "physicalSecurity": "No physical security required",
            "description": "Basic security, production-grade equipment",
            "useCase": "Software-only cryptographic modules",
            "examples": ["OpenSSL FIPS module", "Software crypto libraries"]
        },
        {
            "level": 2,
            "name": "Level 2",
            "physicalSecurity": "Tamper-evident coatings or seals",
            "description": "Role-based authentication, tamper evidence",
            "useCase": "Basic hardware security",
            "examples": ["Tamper-evident HSM enclosures", "Secure boot systems"]
        },
        {
            "level": 3,
            "name": "Level 3",
            "physicalSecurity": "Tamper-resistant with zeroization",
            "description": "Identity-based authentication, key zeroization on tamper",
            "useCase": "CA private key protection, HSMs",
            "examples": ["Hardware Security Modules (HSMs)", "Smart cards"]
        },
        {
            "level": 4,
            "name": "Level 4",
            "physicalSecurity": "Complete envelope of protection",
            "description": "Environmental failure protection, zeroization on any attack",
            "useCase": "High-security environments, military, classified",
            "examples": ["Military-grade HSMs", "Secure government systems"]
        }
    ],
    "timeline": [
        {
            "date": "2019-09-22",
            "event": "FIPS 140-3 becomes effective",
            "impact": "New standard based on ISO/IEC 19790:2012"
        },
        {
            "date": "2021-09-21",
            "event": "FIPS 140-2 testing ended",
            "impact": "No new FIPS 140-2 validations accepted"
        },
        {
            "date": "2026-09-21",
            "event": "FIPS 140-2 certificates expire",
            "impact": "All modules must transition to FIPS 140-3"
        }
    ],
    "caRequirements": {
        "minimumLevel": 3,
        "description": "CA private keys MUST be protected in FIPS 140-2/3 Level 3 or higher HSM",
        "source": "CA/Browser Forum Baseline Requirements",
        "notes": "Root CA keys typically require Level 3; some policies require Level 4"
    }
}

# =============================================================================
# NIST SP 800-57 Part 1 Rev 5 - Recommendation for Key Management
# Source: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf
# =============================================================================
NIST_800_57 = {
    "document": "NIST SP 800-57 Part 1",
    "version": "Revision 5",
    "effectiveDate": "2020-05-01",
    "sourceUrl": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf",
    "lastVerified": "2024-12-15",
    "keyStrengthComparison": [
        {
            "securityBits": 80,
            "status": "disallowed",
            "symmetric": "2TDEA",
            "rsa": 1024,
            "dsa": {"L": 1024, "N": 160},
            "ecc": "160-223",
            "hash": "SHA-1",
            "notes": "No longer approved for federal use"
        },
        {
            "securityBits": 112,
            "status": "legacy",
            "symmetric": "3TDEA",
            "rsa": 2048,
            "dsa": {"L": 2048, "N": 224},
            "ecc": "224-255",
            "hash": "SHA-224/256/384/512",
            "notes": "Acceptable through 2030"
        },
        {
            "securityBits": 128,
            "status": "acceptable",
            "symmetric": "AES-128",
            "rsa": 3072,
            "dsa": {"L": 3072, "N": 256},
            "ecc": "256-383",
            "hash": "SHA-256/384/512",
            "notes": "Recommended minimum for new systems"
        },
        {
            "securityBits": 192,
            "status": "acceptable",
            "symmetric": "AES-192",
            "rsa": 7680,
            "dsa": None,
            "ecc": "384-511",
            "hash": "SHA-384/512",
            "notes": "High security applications"
        },
        {
            "securityBits": 256,
            "status": "acceptable",
            "symmetric": "AES-256",
            "rsa": 15360,
            "dsa": None,
            "ecc": "512+",
            "hash": "SHA-512",
            "notes": "Maximum security level"
        }
    ],
    "cryptoperiods": [
        {
            "keyType": "Private signature key",
            "originatorUsage": "1-3 years",
            "recipientUsage": "Indefinite (for verification)",
            "notes": "Limit active signing period"
        },
        {
            "keyType": "Public signature key",
            "originatorUsage": "N/A",
            "recipientUsage": "Indefinite",
            "notes": "May be needed for verification years later"
        },
        {
            "keyType": "Symmetric key-wrapping key",
            "originatorUsage": "Up to 2 years",
            "recipientUsage": "Up to 3 years",
            "notes": "Protect key distribution"
        },
        {
            "keyType": "Symmetric data encryption key",
            "originatorUsage": "Up to 2 years",
            "recipientUsage": "Up to 3 years",
            "notes": "For data at rest"
        },
        {
            "keyType": "TLS session key",
            "originatorUsage": "Hours to days",
            "recipientUsage": "Hours to days",
            "notes": "Short-lived by design"
        }
    ]
}

# =============================================================================
# NIST SP 800-52 Rev 2 - Guidelines for TLS Implementations
# Source: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-52r2.pdf
# =============================================================================
NIST_800_52 = {
    "document": "NIST SP 800-52",
    "version": "Revision 2",
    "effectiveDate": "2019-08-01",
    "sourceUrl": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-52r2.pdf",
    "lastVerified": "2024-12-15",
    "tlsVersionRequirements": {
        "minimumServer": "TLS 1.2",
        "minimumClient": "TLS 1.2",
        "recommended": "TLS 1.3",
        "prohibited": ["SSL 2.0", "SSL 3.0", "TLS 1.0", "TLS 1.1"],
        "notes": "TLS 1.3 SHOULD be supported; TLS 1.2 MUST be supported"
    },
    "cipherSuites": {
        "tls12Required": [
            "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
        ],
        "tls13Required": [
            "TLS_AES_128_GCM_SHA256",
            "TLS_AES_256_GCM_SHA384"
        ],
        "prohibited": [
            "Any cipher suite with NULL encryption",
            "Any cipher suite with RC4",
            "Any cipher suite with 3DES",
            "Any cipher suite with export-grade cryptography",
            "Any cipher suite with static RSA key exchange"
        ]
    },
    "certificateRequirements": {
        "minimumRsaKeySize": 2048,
        "minimumEccKeySize": 256,
        "requiredHashAlgorithm": "SHA-256 or stronger",
        "prohibitedHashAlgorithms": ["MD5", "SHA-1"],
        "validityPeriod": "Should be as short as practical",
        "revocationChecking": "OCSP or CRL checking SHOULD be performed"
    },
    "serverGuidelines": [
        "Configure servers to prefer ECDHE cipher suites",
        "Disable TLS compression (CRIME attack)",
        "Enable HSTS with long max-age",
        "Support OCSP stapling",
        "Configure secure renegotiation",
        "Disable session tickets if not needed"
    ],
    "clientGuidelines": [
        "Verify server certificates properly",
        "Check certificate revocation status",
        "Reject expired or self-signed certificates in production",
        "Verify hostname matches certificate",
        "Use certificate pinning for high-security applications"
    ]
}

# =============================================================================
# NIST POST-QUANTUM CRYPTOGRAPHY STANDARDS
# Source: https://csrc.nist.gov/projects/post-quantum-cryptography
# =============================================================================
NIST_PQC = {
    "document": "NIST Post-Quantum Cryptography Standards",
    "standards": [
        {
            "fips": "FIPS 203",
            "name": "ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)",
            "publishedDate": "2024-08-13",
            "previousName": "CRYSTALS-Kyber",
            "type": "Key Encapsulation",
            "useCase": "Key exchange, TLS handshake, hybrid key agreement",
            "parameterSets": [
                {"name": "ML-KEM-512", "securityLevel": 1, "publicKeySize": 800, "ciphertextSize": 768},
                {"name": "ML-KEM-768", "securityLevel": 3, "publicKeySize": 1184, "ciphertextSize": 1088},
                {"name": "ML-KEM-1024", "securityLevel": 5, "publicKeySize": 1568, "ciphertextSize": 1568},
            ],
            "notes": "Primary NIST-approved key encapsulation mechanism. Recommended for new deployments."
        },
        {
            "fips": "FIPS 204",
            "name": "ML-DSA (Module-Lattice-Based Digital Signature Algorithm)",
            "publishedDate": "2024-08-13",
            "previousName": "CRYSTALS-Dilithium",
            "type": "Digital Signature",
            "useCase": "Code signing, document signing, certificate signatures",
            "parameterSets": [
                {"name": "ML-DSA-44", "securityLevel": 2, "publicKeySize": 1312, "signatureSize": 2420},
                {"name": "ML-DSA-65", "securityLevel": 3, "publicKeySize": 1952, "signatureSize": 3293},
                {"name": "ML-DSA-87", "securityLevel": 5, "publicKeySize": 2592, "signatureSize": 4595},
            ],
            "notes": "Primary NIST-approved signature algorithm. Approved for CNSA 2.0."
        },
        {
            "fips": "FIPS 205",
            "name": "SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)",
            "publishedDate": "2024-08-13",
            "previousName": "SPHINCS+",
            "type": "Digital Signature",
            "useCase": "Firmware signing, long-term archival signatures, backup to ML-DSA",
            "parameterSets": [
                {"name": "SLH-DSA-128s", "securityLevel": 1, "publicKeySize": 32, "signatureSize": 7856},
                {"name": "SLH-DSA-128f", "securityLevel": 1, "publicKeySize": 32, "signatureSize": 17088},
                {"name": "SLH-DSA-192s", "securityLevel": 3, "publicKeySize": 48, "signatureSize": 16224},
                {"name": "SLH-DSA-256s", "securityLevel": 5, "publicKeySize": 64, "signatureSize": 29792},
            ],
            "notes": "Hash-based signatures. Conservative choice, larger signatures. NOT approved for CNSA 2.0."
        },
    ],
    "upcomingStandards": [
        {
            "name": "FN-DSA (Falcon)",
            "status": "Standardization in progress",
            "expectedDate": "2025",
            "type": "Digital Signature",
            "notes": "Smaller signatures than ML-DSA but more complex implementation. Draft expected 2025."
        },
        {
            "name": "HQC",
            "status": "Standardization in progress", 
            "expectedDate": "2025",
            "type": "Key Encapsulation",
            "notes": "Code-based KEM. Backup to ML-KEM with different mathematical foundation."
        },
    ],
    "cnsa2Approved": {
        "keyEncapsulation": ["ML-KEM-768", "ML-KEM-1024"],
        "digitalSignature": ["ML-DSA-65", "ML-DSA-87"],
        "notes": "CNSA 2.0 does NOT approve SLH-DSA (SPHINCS+) or FN-DSA (Falcon). Use ML-KEM and ML-DSA for NSS compliance."
    },
    "migrationGuidance": {
        "immediate": "Inventory all cryptographic assets. Identify quantum-vulnerable algorithms.",
        "shortTerm": "Enable hybrid modes (classical + PQC) where available. Test PQC in non-production.",
        "mediumTerm": "Deploy PQC for new systems. Plan migration timeline for existing systems.",
        "longTerm": "Complete transition before 2035 NIST deadline.",
        "resources": [
            {"name": "NIST IR 8547", "url": "https://csrc.nist.gov/publications/detail/nistir/8547/final", "description": "Transition to Post-Quantum Cryptography Standards"},
            {"name": "CISA PQC Guidance", "url": "https://www.cisa.gov/quantum", "description": "Post-Quantum Cryptography Initiative"},
            {"name": "NSA CNSA 2.0", "url": "https://media.defense.gov/2022/Sep/07/2003071834/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS_.PDF", "description": "Commercial National Security Algorithm Suite 2.0"},
        ]
    },
    "securityLevelMapping": [
        {"nistLevel": 1, "classicalEquivalent": "AES-128", "minKeyBits": 128},
        {"nistLevel": 2, "classicalEquivalent": "SHA-256", "minKeyBits": 128},
        {"nistLevel": 3, "classicalEquivalent": "AES-192", "minKeyBits": 192},
        {"nistLevel": 4, "classicalEquivalent": "SHA-384", "minKeyBits": 192},
        {"nistLevel": 5, "classicalEquivalent": "AES-256", "minKeyBits": 256},
    ]
}

# =============================================================================
# Cross-Reference: How frameworks align
# =============================================================================
FRAMEWORK_CROSS_REFERENCE = {
    "RSA-2048": {
        "nist800131a": "Acceptable through 2030",
        "cabForum": "Minimum for TLS certificates",
        "pciDss": "Meets 'strong cryptography' requirement",
        "fips140": "Approved algorithm"
    },
    "TLS-1.2": {
        "nist80052": "Minimum required for federal systems",
        "cabForum": "Minimum required",
        "pciDss": "Minimum required",
        "allRootStores": "Required for trust"
    },
    "SHA-256": {
        "nist800131a": "Acceptable, recommended",
        "cabForum": "Required for certificate signatures",
        "pciDss": "Meets strong cryptography",
        "fips140": "Approved algorithm"
    },
    "ECDSA-P256": {
        "nist800131a": "Acceptable (128-bit security)",
        "cabForum": "Approved curve",
        "nist80057": "Recommended",
        "fips140": "Approved algorithm"
    },
    "ML-KEM-768": {
        "nist800131a": "Acceptable (PQC standard)",
        "cabForum": "Not yet required, monitoring",
        "pciDss": "Not yet addressed",
        "fips140": "FIPS 203 approved",
        "cnsa2": "Approved for NSS"
    },
    "ML-DSA-65": {
        "nist800131a": "Acceptable (PQC standard)",
        "cabForum": "Not yet required, monitoring",
        "pciDss": "Not yet addressed",
        "fips140": "FIPS 204 approved",
        "cnsa2": "Approved for NSS"
    }
}

# =============================================================================
# PKI NEWS RSS FEEDS
# Aggregated news from industry sources
# Last verified: 2025-12-30
# =============================================================================

NEWS_FEEDS = [
    {"id": "google_security", "name": "Google Security Blog", "url": "https://security.googleblog.com/feeds/posts/default", "category": "browser_updates", "icon": "google"},
    {"id": "mozilla_security", "name": "Mozilla Security Blog", "url": "https://blog.mozilla.org/security/feed/", "category": "browser_updates", "icon": "mozilla"},
    {"id": "letsencrypt", "name": "Let's Encrypt", "url": "https://letsencrypt.org/feed.xml", "category": "ca_news", "icon": "letsencrypt"},
    {"id": "digicert", "name": "DigiCert Blog", "url": "https://www.digicert.com/blog/feed", "category": "ca_news", "icon": "digicert"},
    {"id": "sectigo", "name": "Sectigo Blog", "url": "https://sectigo.com/resource-library/rss", "category": "ca_news", "icon": "sectigo"},
    {"id": "globalsign", "name": "GlobalSign Blog", "url": "https://www.globalsign.com/en/blog/rss", "category": "ca_news", "icon": "globalsign"},
    {"id": "sslcom", "name": "SSL.com Blog", "url": "https://www.ssl.com/feed/", "category": "ca_news", "icon": "sslcom"},
    {"id": "hashicorp", "name": "HashiCorp Blog", "url": "https://www.hashicorp.com/blog/feed.xml", "category": "vendors", "icon": "hashicorp"},
    {"id": "qualys_ssl", "name": "Qualys Security Blog", "url": "https://blog.qualys.com/feed", "category": "vendors", "icon": "qualys"},
    {"id": "cryptoeng", "name": "Cryptography Engineering", "url": "https://blog.cryptographyengineering.com/feed/", "category": "research", "icon": "research"},
]

PKI_KEYWORDS = [
    'certificate', 'ssl', 'tls', 'pki', 'x.509', 'x509',
    'certificate authority', 'https', 'encryption',
    'root program', 'distrust', 'revocation', 'ocsp', 'crl',
    'acme', 'ballot', 'baseline requirements',
    'key compromise', 'mis-issuance', 'webpki', 'web pki',
    'cipher', 'cryptography', 'rsa', 'ecc', 'ecdsa',
    'post-quantum', 'pqc', 'validity period', 'expiration',
    'ca/browser', 'cabforum', 'root store', 'intermediate',
    'key length', 'sha-256', 'sha256', 'signing', 'verification'
]

PRIORITY_KEYWORDS = [
    'distrust', 'revocation', 'ballot passed', 'ballot failed',
    'deadline', 'security incident', 'vulnerability',
    'chrome root', 'mozilla root', 'apple root', 'microsoft root',
    'mis-issuance', 'compromise', 'urgent', 'breaking change',
    'deprecation', 'sunset', 'end of life', 'mandatory'
]

NEWS_FILE = DATA_DIR / "news.json"
NEWS_STALE_HOURS = 6
NEWS_MAX_AGE_DAYS = 90

def load_news() -> Dict[str, Any]:
    """Load cached news items."""
    ensure_data_dir()
    if NEWS_FILE.exists():
        try:
            return json.loads(NEWS_FILE.read_text())
        except json.JSONDecodeError:
            return {"items": [], "lastFetched": None}
    return {"items": [], "lastFetched": None}

def save_news(news_data: Dict[str, Any]):
    """Save news items to disk."""
    ensure_data_dir()
    NEWS_FILE.write_text(json.dumps(news_data, indent=2, default=str))

def is_news_stale() -> bool:
    """Check if news data needs refreshing."""
    news_data = load_news()
    if not news_data.get("lastFetched"):
        return True
    try:
        last_fetched = datetime.fromisoformat(news_data["lastFetched"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - last_fetched).total_seconds() / 3600
        return age_hours > NEWS_STALE_HOURS
    except:
        return True

def passes_keyword_filter(title: str, description: str) -> bool:
    """Check if content contains PKI-related keywords."""
    text = (title + ' ' + description).lower()
    return any(keyword in text for keyword in PKI_KEYWORDS)

def is_priority_item(title: str, description: str) -> bool:
    """Check if content contains priority keywords."""
    text = (title + ' ' + description).lower()
    return any(keyword in text for keyword in PRIORITY_KEYWORDS)

def parse_rss_feed(feed_config: Dict[str, str]) -> List[Dict[str, Any]]:
    """Parse a single RSS/Atom feed and filter for PKI content."""
    import uuid
    import re
    try:
        from bs4 import BeautifulSoup
        HAS_BS4 = True
    except ImportError:
        HAS_BS4 = False
    
    items = []
    try:
        parsed = feedparser.parse(feed_config["url"])
        for entry in parsed.entries[:20]:
            title = entry.get("title", "")
            description = entry.get("summary", entry.get("description", ""))
            
            if not passes_keyword_filter(title, description):
                continue
            
            if HAS_BS4:
                soup = BeautifulSoup(description, "html.parser")
                clean_desc = soup.get_text()[:300].strip()
            else:
                clean_desc = re.sub(r'<[^>]+>', '', description)[:300].strip()
            if len(clean_desc) > 297:
                clean_desc = clean_desc[:297] + "..."
            
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except:
                    published = datetime.now(timezone.utc).isoformat()
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).isoformat()
                except:
                    published = datetime.now(timezone.utc).isoformat()
            else:
                published = datetime.now(timezone.utc).isoformat()
            
            items.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "url": entry.get("link", ""),
                "source": feed_config["name"],
                "sourceId": feed_config["id"],
                "sourceUrl": feed_config["url"].rsplit("/", 1)[0] if "/" in feed_config["url"] else feed_config["url"],
                "category": feed_config["category"],
                "icon": feed_config.get("icon", "default"),
                "excerpt": clean_desc,
                "publishedAt": published,
                "fetchedAt": datetime.now(timezone.utc).isoformat(),
                "isPriority": is_priority_item(title, description)
            })
    except Exception as e:
        print(f"[NEWS] Error fetching {feed_config['name']}: {e}")
    
    return items

def fetch_all_news_feeds() -> Dict[str, Any]:
    """Fetch all RSS feeds and aggregate news items."""
    all_items = []
    
    for feed_config in NEWS_FEEDS:
        items = parse_rss_feed(feed_config)
        all_items.extend(items)
    
    existing_news = load_news()
    existing_urls = {item["url"] for item in existing_news.get("items", [])}
    
    new_items = [item for item in all_items if item["url"] not in existing_urls]
    combined_items = new_items + existing_news.get("items", [])
    
    cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=NEWS_MAX_AGE_DAYS)
    filtered_items = []
    for item in combined_items:
        try:
            pub_date = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
            if pub_date > cutoff:
                filtered_items.append(item)
        except:
            filtered_items.append(item)
    
    filtered_items.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
    
    news_data = {
        "items": filtered_items[:200],
        "lastFetched": datetime.now(timezone.utc).isoformat(),
        "feedsChecked": len(NEWS_FEEDS),
        "newItemsAdded": len(new_items)
    }
    
    save_news(news_data)
    return news_data

def get_news(category: Optional[str] = None, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Get news items with optional filtering."""
    if is_news_stale():
        fetch_all_news_feeds()
    
    news_data = load_news()
    items = news_data.get("items", [])
    
    if category and category != "all":
        items = [item for item in items if item.get("category") == category]
    
    total = len(items)
    paginated_items = items[offset:offset + limit]
    
    return {
        "items": paginated_items,
        "total": total,
        "hasMore": offset + limit < total,
        "updatedAt": news_data.get("lastFetched"),
        "offset": offset,
        "limit": limit
    }

def get_news_sources() -> Dict[str, Any]:
    """Get list of all news feed sources."""
    sources = []
    for feed in NEWS_FEEDS:
        sources.append({
            "id": feed["id"],
            "name": feed["name"],
            "url": feed["url"].rsplit("/", 1)[0] if "/" in feed["url"] else feed["url"],
            "category": feed["category"],
            "icon": feed.get("icon", "default")
        })
    return {"sources": sources}

def is_data_stale():
    """Check if compliance data is stale based on last review date."""
    from datetime import timedelta
    last_review = datetime.strptime(DATA_FRESHNESS["lastFullReview"], "%Y-%m-%d")
    stale_threshold = timedelta(days=DATA_FRESHNESS["staleAfterDays"])
    return datetime.now() > last_review + stale_threshold

# ============================================================================
# State Management
# ============================================================================

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_state() -> Dict[str, Any]:
    """Load persisted state from disk."""
    ensure_data_dir()
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_check": None,
        "feed_states": {},
        "document_hashes": {},
        "seen_items": [],
    }

def save_state(state: Dict[str, Any]):
    """Save state to disk."""
    ensure_data_dir()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

def load_cache() -> Dict[str, Any]:
    """Load cached content."""
    ensure_data_dir()
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {"feeds": {}, "documents": {}}

def save_cache(cache: Dict[str, Any]):
    """Save cache to disk."""
    ensure_data_dir()
    CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))

# ============================================================================
# Pydantic Models
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class CheckFeedsInput(BaseModel):
    """Input for checking RSS/Atom feeds for updates."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    feed_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific feed IDs to check. If None, checks all feeds. "
                    "Available: cabforum_public, ccadb_public, chrome_security_blog, mozilla_security"
    )
    since_days: int = Field(
        default=7,
        description="Only return items from the last N days",
        ge=1,
        le=90
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured"
    )

class CheckDocumentInput(BaseModel):
    """Input for checking a specific document for changes."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(
        ...,
        description="Document ID to check. Available: cabf_br, chrome_root_policy, "
                    "mozilla_root_policy, apple_root_program, microsoft_root_program, "
                    "microsoft_root_announcements, microsoft_release_notes, "
                    "nist_800_131a, nist_800_57"
    )
    persist: bool = Field(
        default=True,
        description="Write the new hash to state. TRUE (default) makes this a "
                    "CONSUMING read: the next caller will see 'unchanged'. Pass "
                    "False to peek without consuming the change for other callers."
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class CheckAllDocumentsInput(BaseModel):
    """Input for checking all tracked documents."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    priority: Optional[str] = Field(
        default=None,
        description="Filter by priority: 'high', 'medium', or None for all"
    )
    persist: bool = Field(
        default=True,
        description="Write the new hashes to state. TRUE (default) makes this a "
                    "CONSUMING read: the next caller will see 'unchanged'. Pass "
                    "False to peek without consuming changes for other callers."
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class GetDeadlinesInput(BaseModel):
    """Input for getting compliance deadlines."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    category: Optional[str] = Field(
        default=None,
        description="Filter by category: 'validity', 'validation', 'algorithm', 'enforcement', 'reporting', 'documentation', 'legislative', 'testing', 'technical_standards', or None for all"
    )
    within_days: Optional[int] = Field(
        default=None,
        description="Only show deadlines within N days from now",
        ge=1,
        le=3650
    )
    framework: Optional[str] = Field(
        default=None,
        description="Filter by framework: 'cabforum', 'dora', 'nis2', 'uk-csr', or None for all"
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Filter by jurisdiction: 'global', 'eu', 'uk', 'us', or None for all"
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: 'passed', 'ongoing', 'upcoming', or None for all"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class GetFrameworksInput(BaseModel):
    """Input for getting regulatory frameworks."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    jurisdiction: Optional[str] = Field(
        default=None,
        description="Filter by jurisdiction: 'global', 'eu', 'uk', 'us', or None for all"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class GetFrameworkInput(BaseModel):
    """Input for getting a single framework with deadlines."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    framework_id: str = Field(
        ...,
        description="Framework ID: 'cabforum', 'dora', 'nis2', 'uk-csr'"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class SearchCCADBInput(BaseModel):
    """Input for searching CCADB discussions."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Search terms (e.g., 'Entrust', 'revocation', 'incident')",
        min_length=2,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class GetStatusInput(BaseModel):
    """Input for getting monitoring status."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class FetchDocumentInput(BaseModel):
    """Input for fetching full document content."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(
        ...,
        description="Document ID to fetch"
    )

class ListContentCandidatesInput(BaseModel):
    """Input for listing content candidates."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    sink_status: Optional[str] = Field(
        default=None,
        description="Filter by sink status: 'pending' (awaiting news-desk delivery) or 'posted'"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max candidates to return (newest first)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_url(url: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch URL content with error handling."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "PKI-Compliance-Monitor/1.0 (FixMyCert.com)"
            })
            response.raise_for_status()
            return response.text
    except Exception as e:
        return None

# Per-request dynamic content that makes document hashes churn without any
# real document change (seen daily on csrc.nist.gov behind Cloudflare):
# challenge <script> params, email-protection href tokens, data-cfemail attrs.
_HASH_NOISE_PATTERNS = [
    re.compile(r"<script\b.*?</script>", re.DOTALL | re.IGNORECASE),
    re.compile(r"/cdn-cgi/l/email-protection#[0-9a-f]+"),
    re.compile(r'data-cfemail="[0-9a-f]+"'),
]

# learn.microsoft.com page chrome that re-renders per request (false positive
# on microsoft_root_program 2026-07-21): the "AI Summary" block regenerates,
# and the "Additional resources"/Events right-rail rotates its events daily.
# Any Learn page we track carries both. Each entry names the element tag and
# a pattern for its opening tag; the region through the balanced closing tag
# is removed — these blocks nest same-name tags, so a lazy regex would stop
# at the first inner close tag and leave the dynamic tail in the hash.
_HASH_NOISE_BLOCKS = [
    ("div", re.compile(r'<div\b[^>]*\bdata-id="ai-summary"[^>]*>', re.IGNORECASE)),
    ("div", re.compile(r'<div\b[^>]*\bid="ms--ai-summary-header"[^>]*>', re.IGNORECASE)),
    ("div", re.compile(r'<div\b[^>]*\bid="ms--additional-resources(?:-mobile)?"[^>]*>', re.IGNORECASE)),
    ("section", re.compile(
        r'<section\b[^>]*\bdata-bi-name="(?:events-card|recommendations|learning-resource-card)"[^>]*>',
        re.IGNORECASE)),
]


# csrc.nist.gov publication pages (nist_800_57, nist_800_131a) are the only
# tracked documents that hash a rendered HTML page whole — every other page
# source was repointed to a raw artifact or scoped by check_url. Their site
# chrome (nav, footer, banners) can be redeployed with no publication event,
# which is what moved nist_800_57 on 2026-08-07 and nist_800_131a on
# 2026-08-12: both hashes were stable across back-to-back fetches yet had
# moved with nothing published behind them. When one of these containers is
# present, only its region is hashed. The publications-detail panel carries
# everything that constitutes a publication event — Date Published, Planning
# Note, Document History, Supersedes — so the revision/withdrawal signal that
# argued against repointing to the PDF artifact is preserved.
_HASH_CONTENT_REGIONS = [
    ("div", re.compile(
        r'<div\b[^>]*\bclass="[^"]*\bpublications-detail\b[^"]*"[^>]*>',
        re.IGNORECASE)),
]


def _balanced_end(content: str, tag: str, m: "re.Match") -> int:
    """End offset of the balanced </tag> for the block opened at match m.

    If the markup is unbalanced (no matching close tag), returns m.end() so
    callers degrade to the opening tag alone and always terminate.
    """
    tag_pat = re.compile(rf"<(/?){tag}\b[^>]*>", re.IGNORECASE)
    depth = 1
    for t in tag_pat.finditer(content, m.end()):
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return t.end()
    return m.end()


def _strip_balanced_blocks(content: str, tag: str, start_pat: "re.Pattern") -> str:
    """Remove every region from a start_pat match through its balanced </tag>."""
    while True:
        m = start_pat.search(content)
        if not m:
            return content
        content = content[:m.start()] + content[_balanced_end(content, tag, m):]


def hash_content(content: str) -> str:
    """Generate SHA-256 hash of content, ignoring per-request dynamic noise."""
    # Region scoping first, on the raw page: a chrome change outside the
    # region must not move the hash, and the noise rules below still apply
    # inside the extracted region.
    for tag, start_pat in _HASH_CONTENT_REGIONS:
        m = start_pat.search(content)
        if m:
            content = content[m.start():_balanced_end(content, tag, m)]
            break
    # Scripts next: script bodies may contain literal tag text that would
    # otherwise confuse the balanced-block scan.
    for pat in _HASH_NOISE_PATTERNS:
        content = pat.sub("", content)
    for tag, start_pat in _HASH_NOISE_BLOCKS:
        content = _strip_balanced_blocks(content, tag, start_pat)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def parse_feed(content: str) -> List[Dict[str, Any]]:
    """Parse RSS/Atom feed content."""
    feed = feedparser.parse(content)
    items = []
    for entry in feed.entries:
        items.append({
            "title": entry.get("title", "No title"),
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "")),
            "summary": entry.get("summary", "")[:500],
        })
    return items

def format_datetime(dt_str: str) -> str:
    """Format datetime string for display."""
    try:
        # Try various formats
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d"]:
            try:
                dt = datetime.strptime(dt_str, fmt)
                return dt.strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                continue
        return dt_str
    except Exception:
        return dt_str

def days_until(date_str: str) -> int:
    """Whole days from today (UTC) to a date. Negative once the date is past.

    CALENDAR-DATE arithmetic, deliberately — not a timestamp difference.
    This used to be `(target_midnight - now).days`, and timedelta.days floors
    toward negative infinity, so a partial day always resolved AWAY from
    today: 2026-08-01 read as -15 at 13:19 UTC on 2026-08-15 when 14 days had
    elapsed, and a date 170 calendar days out read as 169. Worse than the
    label, it made a deadline compute "passed" on its own due date, from
    00:00:01 UTC onward — green "Completed" on the morning it actually bites.
    Comparing dates gives 0 on the day itself, -1 the day after. (2026-08-15)
    """
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    return (target - today).days


# Stable column order for the deadlines CSV export. Nested feed fields are
# flattened: relatedGuides -> "; "-joined URLs, consequences -> two columns.
DEADLINE_CSV_COLUMNS = [
    "date", "days_until", "status", "title", "source", "framework_name",
    "category", "jurisdiction", "is_major", "is_estimated", "impact",
    "description", "source_url", "related_guides", "note",
    "consequence_enforcement", "consequence_scenario",
]


def build_deadlines_csv(
    category: Optional[str] = None,
    framework: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    status: Optional[str] = None,
    within_days: Optional[int] = None,
) -> str:
    """Render unified deadlines as CSV text (header + rows, sorted by date).

    Applies the same filters as get_deadlines(). Uses the csv module so
    commas/quotes/newlines in descriptions are correctly escaped. Returns a
    str; the HTTP layer encodes it utf-8-sig so Excel renders accented and
    em-dash characters. Column order is DEADLINE_CSV_COLUMNS.
    """
    rows = []
    for d in get_all_deadlines_unified():
        if category and d.get("category") != category:
            continue
        if framework and d.get("framework_id") != framework:
            continue
        if jurisdiction and d.get("jurisdiction") != jurisdiction:
            continue
        if status and d.get("status") != status:
            continue
        days = days_until(d["date"])
        if within_days is not None and days > within_days:
            continue
        cons = d.get("consequences") or {}
        rows.append({
            "date": d["date"],
            "days_until": days,
            "status": d.get("status", ""),
            "title": d.get("title", ""),
            "source": d.get("source", ""),
            "framework_name": d.get("framework_name", ""),
            "category": d.get("category", ""),
            "jurisdiction": d.get("jurisdiction", ""),
            "is_major": "true" if d.get("isMajor") else "false",
            "is_estimated": "true" if d.get("is_estimated") else "false",
            "impact": d.get("impact", ""),
            "description": d.get("description", ""),
            "source_url": d.get("source_url") or "",
            "related_guides": "; ".join(g.get("url", "") for g in d.get("relatedGuides", [])),
            "note": d.get("note", ""),
            "consequence_enforcement": cons.get("enforcement", ""),
            "consequence_scenario": cons.get("scenario", ""),
        })
    rows.sort(key=lambda r: r["date"])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=DEADLINE_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def calculate_status(date_str: str, current_status: Optional[str] = None) -> str:
    """Calculate deadline status based on date.
    
    If status is 'ongoing', it stays ongoing regardless of date.
    Otherwise, calculates based on whether date is in the past. A deadline
    due TODAY computes "upcoming", not "passed" — it has not passed until
    the day is over (fixed with days_until, 2026-08-15).
    """
    if current_status == "ongoing":
        return "ongoing"
    
    days = days_until(date_str)
    if days < 0:
        return "passed"
    return "upcoming"


def get_all_deadlines_unified() -> List[Dict[str, Any]]:
    """Get all deadlines from DEADLINES and REGULATORY_FRAMEWORKS combined.
    
    Returns unified list with framework_id, jurisdiction, and calculated status.
    """
    all_deadlines = []
    
    # Heuristic defaults applied only when the entry doesn't set the field
    # explicitly. Putting defaults BEFORE `**d` means entry values win on the
    # merge — the prior order (defaults after `**d`) silently clobbered
    # entries that set jurisdiction/framework_id/status themselves.
    framework_map = {
        "cab-forum": "cabforum",
        "chrome": "cabforum",
        "mozilla": "cabforum",
        "apple": "cabforum",
        "microsoft": "cabforum",
        "nist": "nist",
        "nsa": "nsa",
    }

    # Frameworks with an on-site guide (resource_link under /guides/) lend it
    # to their deadlines as a relatedGuides fallback — used when neither the
    # entry nor its category provides guides.
    framework_guides_by_id = {}
    for framework in REGULATORY_FRAMEWORKS:
        link = framework.get("resource_link")
        if link and link.startswith("/guides/"):
            framework_guides_by_id[framework["framework_id"]] = [{
                "title": f"{framework['name']} Certificate Management",
                "url": link,
                "hasVideo": False,
            }]

    for d in DEADLINES:
        source = d.get("source", "unknown")
        default_jurisdiction = "us" if source in ("nist", "nsa") else "global"
        default_framework_id = framework_map.get(source, None)

        # ballotStatus tracks the BALLOT (proposed/voting/passed), not the
        # deadline: a passed ballot with a future effective date is still an
        # upcoming deadline. Only "ongoing" bypasses date-based status.
        ballot_status = d.get("ballotStatus", "upcoming")
        if ballot_status in ("proposed", "voting", "passed"):
            ballot_status = "upcoming"
        default_status = (
            ballot_status if ballot_status == "ongoing"
            else calculate_status(d["date"], ballot_status)
        )

        entry = {
            "framework_id": default_framework_id,
            "jurisdiction": default_jurisdiction,
            "status": default_status,
            "source_url": None,
            **d,  # entry-level fields override heuristic defaults
        }
        if "relatedGuides" not in entry:
            entry["relatedGuides"] = (
                EXPLICIT_RELATED_GUIDES.get(entry["id"])
                or CATEGORY_RELATED_GUIDES.get(entry.get("category"))
                or framework_guides_by_id.get(entry.get("framework_id"))
                or []
            )
        all_deadlines.append(entry)

    for framework in REGULATORY_FRAMEWORKS:
        # Framework deadline categories (effective, reporting, registration)
        # are framework-generic, so the framework's own guide is usually what
        # the category map falls through to here.
        framework_guides = framework_guides_by_id.get(framework["framework_id"], [])
        for deadline in framework.get("deadlines", []):
            all_deadlines.append({
                "framework_id": framework["framework_id"],
                "framework_name": framework["name"],
                "jurisdiction": framework["jurisdiction"],
                "source": framework["framework_id"],
                "status": calculate_status(deadline["date"], deadline.get("status")),
                "isMajor": deadline.get("impact") == "high",
                "source_url": None,
                "relatedGuides": (
                    EXPLICIT_RELATED_GUIDES.get(deadline["id"])
                    or CATEGORY_RELATED_GUIDES.get(deadline.get("category"))
                    or framework_guides
                ),
                **deadline,  # entry-level fields override framework-level defaults
            })
    
    return all_deadlines


def get_frameworks_list(jurisdiction: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get list of regulatory frameworks with metadata (no deadlines).
    
    Optionally filter by jurisdiction. Counts deadlines from unified list.
    """
    all_unified = get_all_deadlines_unified()
    
    frameworks = []
    for f in REGULATORY_FRAMEWORKS:
        if jurisdiction and f["jurisdiction"] != jurisdiction:
            continue
        
        framework_deadlines = [d for d in all_unified if d.get("framework_id") == f["framework_id"]]
        upcoming_deadlines = [d for d in framework_deadlines if d.get("status") == "upcoming"]
        
        next_deadline = None
        if upcoming_deadlines:
            upcoming_deadlines.sort(key=lambda x: x["date"])
            next_deadline = upcoming_deadlines[0]["date"]
        
        frameworks.append({
            "framework_id": f["framework_id"],
            "name": f["name"],
            "full_name": f["full_name"],
            "jurisdiction": f["jurisdiction"],
            "effective_date": f["effective_date"],
            "description": f["description"],
            "applies_to": f["applies_to"],
            "certificate_relevance": f["certificate_relevance"],
            "resource_link": f["resource_link"],
            "deadline_count": len(framework_deadlines),
            "upcoming_count": len(upcoming_deadlines),
            "next_deadline": next_deadline,
        })
    
    return frameworks


def get_framework_by_id(framework_id: str) -> Optional[Dict[str, Any]]:
    """Get a single framework with all its deadlines from unified list."""
    for f in REGULATORY_FRAMEWORKS:
        if f["framework_id"] == framework_id:
            all_unified = get_all_deadlines_unified()
            framework_deadlines = [d for d in all_unified if d.get("framework_id") == framework_id]
            
            deadlines_with_status = []
            for d in framework_deadlines:
                deadlines_with_status.append({
                    **d,
                    "days_until": days_until(d["date"]),
                    "is_past": days_until(d["date"]) < 0,
                })
            deadlines_with_status.sort(key=lambda x: x["date"])
            
            return {
                **f,
                "deadlines": deadlines_with_status,
            }
    return None


# ============================================================================
# MCP Server (optional - for native MCP clients)
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("pki_compliance_mcp")

    def mcp_tool(**kwargs):
        """Decorator for MCP tools."""
        return mcp.tool(**kwargs)
else:
    mcp = None

    def mcp_tool(**kwargs):
        """No-op decorator when MCP is not available."""
        def decorator(func):
            return func
        return decorator


@mcp_tool(
    name="pki_check_feeds",
    annotations={
        "title": "Check PKI Compliance Feeds",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def check_feeds(params: CheckFeedsInput) -> str:
    """Check RSS/Atom feeds for PKI compliance updates.

    Monitors feeds from CA/Browser Forum, CCADB, Chrome Security Blog, 
    and Mozilla Security Blog for recent posts about PKI changes.

    Args:
        params: CheckFeedsInput with feed_ids, since_days, and response_format

    Returns:
        Recent feed items in markdown or JSON format
    """
    state = load_state()
    cache = load_cache()
    results = []

    feed_ids = params.feed_ids or list(FEEDS.keys())

    for feed_id in feed_ids:
        if feed_id not in FEEDS:
            continue

        feed_info = FEEDS[feed_id]

        # Skip non-RSS feeds for now (Google Groups requires scraping)
        if feed_info["type"] in ["google_group", "mailman_archive"]:
            results.append({
                "feed_id": feed_id,
                "name": feed_info["name"],
                "status": "manual_check_required",
                "url": feed_info["url"],
                "items": [],
            })
            continue

        content = await fetch_url(feed_info["url"])
        if not content:
            results.append({
                "feed_id": feed_id,
                "name": feed_info["name"],
                "status": "fetch_failed",
                "items": [],
            })
            continue

        items = parse_feed(content)

        # Filter by date (simplified - in production would parse dates properly)
        results.append({
            "feed_id": feed_id,
            "name": feed_info["name"],
            "status": "ok",
            "item_count": len(items),
            "items": items[:10],  # Return top 10
        })

    # Update state
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "feeds": results,
            "manualCheckRequired": MANUAL_CHECK_REQUIRED,
            "checked_at": state["last_check"]
        }, indent=2)

    # Markdown format
    lines = ["# PKI Compliance Feed Check\n"]
    lines.append(f"**Checked:** {state['last_check']}\n")

    for feed in results:
        lines.append(f"\n## {feed['name']}")
        lines.append(f"Status: `{feed['status']}`\n")

        if feed["status"] == "manual_check_required":
            lines.append(f"→ [Check manually]({feed.get('url', '')})\n")
            continue

        if feed["status"] == "fetch_failed":
            lines.append("⚠️ Could not fetch feed\n")
            continue

        if not feed["items"]:
            lines.append("No recent items\n")
            continue

        for item in feed["items"][:5]:
            title = item["title"][:80]
            lines.append(f"- **[{title}]({item['link']})**")
            if item["published"]:
                lines.append(f"  - {format_datetime(item['published'])}")

    # Add manual check required section
    if MANUAL_CHECK_REQUIRED:
        lines.append("\n---\n## ⚠️ Manual Check Required\n")
        lines.append("The following sources do not have RSS feeds and require manual checking:\n")
        for source in MANUAL_CHECK_REQUIRED:
            lines.append(f"### [{source['name']}]({source['url']})")
            lines.append(f"- **Check frequency:** {source['check_frequency']}")
            lines.append(f"- **Priority:** {source['priority']}")
            lines.append(f"- {source['description']}\n")

    return "\n".join(lines)

@mcp_tool(
    name="pki_check_document",
    annotations={
        "title": "Check Document for Changes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def check_document(params: CheckDocumentInput) -> str:
    """Check a specific PKI compliance document for changes.

    Fetches the document page and compares its hash to the last known version
    to detect updates.

    Args:
        params: CheckDocumentInput with document_id and response_format

    Returns:
        Document status and change detection results
    """
    if params.document_id not in DOCUMENTS:
        available = ", ".join(DOCUMENTS.keys())
        return f"Error: Unknown document_id '{params.document_id}'. Available: {available}"

    doc_info = DOCUMENTS[params.document_id]
    state = load_state()

    url = doc_info.get("check_url", doc_info["url"])
    content = await fetch_url(url)

    if not content:
        return f"Error: Could not fetch {doc_info['name']} from {url}"

    current_hash = hash_content(content)
    previous_hash = state.get("document_hashes", {}).get(params.document_id)

    changed = previous_hash is not None and current_hash != previous_hash
    is_new = previous_hash is None

    # Update state
    if "document_hashes" not in state:
        state["document_hashes"] = {}
    state["document_hashes"][params.document_id] = current_hash
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    # See the note in check_all_documents: saving is what makes this consuming.
    if params.persist:
        save_state(state)

    result = {
        "document_id": params.document_id,
        "name": doc_info["name"],
        "url": doc_info["url"],
        "priority": doc_info["priority"],
        "current_hash": current_hash,
        "previous_hash": previous_hash,
        "changed": changed,
        "is_new": is_new,
        "checked_at": state["last_check"],
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    # Markdown format
    status_emoji = "🔴 CHANGED" if changed else ("🆕 First check" if is_new else "✅ No changes")

    lines = [
        f"# {doc_info['name']}",
        f"\n**Status:** {status_emoji}",
        f"**Priority:** {doc_info['priority']}",
        f"**URL:** {doc_info['url']}",
        f"**Hash:** `{current_hash}`",
        f"**Checked:** {state['last_check']}",
    ]

    if changed:
        lines.append(f"\n⚠️ **Document has changed since last check!**")
        lines.append(f"Previous hash: `{previous_hash}`")
        lines.append(f"\nReview the document for updates to compliance requirements.")

    return "\n".join(lines)

@mcp_tool(
    name="pki_check_all_documents",
    annotations={
        "title": "Check All Tracked Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def check_all_documents(params: CheckAllDocumentsInput) -> str:
    """Check all tracked PKI compliance documents for changes.

    Iterates through all configured documents and reports which ones
    have changed since the last check.

    Args:
        params: CheckAllDocumentsInput with priority filter and response_format

    Returns:
        Summary of all document statuses
    """
    state = load_state()
    results = []

    for doc_id, doc_info in DOCUMENTS.items():
        if params.priority and doc_info["priority"] != params.priority:
            continue

        url = doc_info.get("check_url", doc_info["url"])
        content = await fetch_url(url)

        if not content:
            results.append({
                "document_id": doc_id,
                "name": doc_info["name"],
                "status": "fetch_failed",
                "changed": False,
            })
            continue

        current_hash = hash_content(content)
        previous_hash = state.get("document_hashes", {}).get(doc_id)

        changed = previous_hash is not None and current_hash != previous_hash
        is_new = previous_hash is None

        # Update state
        if "document_hashes" not in state:
            state["document_hashes"] = {}
        state["document_hashes"][doc_id] = current_hash

        results.append({
            "document_id": doc_id,
            "name": doc_info["name"],
            "priority": doc_info["priority"],
            "url": doc_info["url"],
            "status": "changed" if changed else ("new" if is_new else "unchanged"),
            "changed": changed,
            "is_new": is_new,
            "hash": current_hash,
        })

    state["last_check"] = datetime.now(timezone.utc).isoformat()
    # Persisting is what makes this a CONSUMING read: once the new hashes are
    # written, the NEXT caller sees "unchanged". Two schedulers call this 30
    # minutes apart on the same container — the 10:00 research gate (via
    # docker exec, compliance_auto_refresh._DETECT_SNIPPET) and then the 10:30
    # daily_doc_check — so an unconditional save meant the gate silently ate
    # every change before the doc-check could report it to the daily email.
    # Verified twice: 2026-08-02 (gate fired on apple_root_program, that
    # morning's doc-check recorded changes_detected: 0) and 2026-08-06 (gate
    # fired on microsoft_root_announcements, doc-check reported it unchanged
    # at the post-change hash). Readers pass persist=False; only the scheduled
    # doc-check should consume.
    if params.persist:
        save_state(state)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "documents": results,
            "checked_at": state["last_check"],
            "changes_detected": sum(1 for r in results if r["changed"]),
        }, indent=2)

    # Markdown format
    changes = [r for r in results if r["changed"]]

    lines = ["# PKI Document Status Check\n"]
    lines.append(f"**Checked:** {state['last_check']}")
    lines.append(f"**Documents checked:** {len(results)}")
    lines.append(f"**Changes detected:** {len(changes)}\n")

    if changes:
        lines.append("## 🔴 Changed Documents\n")
        for doc in changes:
            lines.append(f"- **{doc['name']}** ({doc['priority']} priority)")
            lines.append(f"  - [View document]({doc['url']})")

    lines.append("\n## All Documents\n")
    lines.append("| Document | Priority | Status |")
    lines.append("|----------|----------|--------|")

    for doc in results:
        status = "🔴 Changed" if doc["changed"] else ("🆕 New" if doc.get("is_new") else "✅ OK")
        if doc["status"] == "fetch_failed":
            status = "⚠️ Failed"
        lines.append(f"| {doc['name']} | {doc.get('priority', 'unknown')} | {status} |")

    return "\n".join(lines)

@mcp_tool(
    name="pki_get_deadlines",
    annotations={
        "title": "Get Compliance Deadlines",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_deadlines(params: GetDeadlinesInput) -> str:
    """Get upcoming PKI compliance deadlines.

    Returns a list of known compliance deadlines from CA/Browser Forum,
    browser root programs, regulatory frameworks (DORA, NIS2, UK CSR), and other PKI standards bodies.

    Args:
        params: GetDeadlinesInput with category, within_days, framework, jurisdiction, status, and response_format

    Returns:
        List of deadlines sorted by date
    """
    now = datetime.now(timezone.utc)

    all_unified = get_all_deadlines_unified()
    deadlines = []
    
    for d in all_unified:
        if params.category and d.get("category") != params.category:
            continue
        
        if params.framework and d.get("framework_id") != params.framework:
            continue
        
        if params.jurisdiction and d.get("jurisdiction") != params.jurisdiction:
            continue
        
        if params.status and d.get("status") != params.status:
            continue

        days = days_until(d["date"])

        if params.within_days and days > params.within_days:
            continue

        deadlines.append({
            **d,
            "days_until": days,
            "is_past": days < 0,
        })

    deadlines.sort(key=lambda x: x["date"])

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "deadlines": deadlines,
            "as_of": now.isoformat(),
            "filters": {
                "category": params.category,
                "framework": params.framework,
                "jurisdiction": params.jurisdiction,
                "status": params.status,
                "within_days": params.within_days,
            }
        }, indent=2)

    lines = ["# PKI Compliance Deadlines\n"]
    lines.append(f"**As of:** {now.strftime('%Y-%m-%d')}\n")

    upcoming = [d for d in deadlines if not d["is_past"]]
    past = [d for d in deadlines if d["is_past"]]

    if upcoming:
        lines.append("## Upcoming Deadlines\n")
        for d in upcoming:
            urgency = "🔴" if d["days_until"] < 90 else ("🟡" if d["days_until"] < 365 else "🟢")
            lines.append(f"### {urgency} {d['title']}")
            lines.append(f"**Date:** {d['date']} ({d['days_until']} days)")
            lines.append(f"**Source:** {d.get('source', d.get('framework_id', 'unknown'))}")
            lines.append(f"**Category:** {d.get('category', 'general')}")
            if d.get('framework_name'):
                lines.append(f"**Framework:** {d['framework_name']}")
            lines.append(f"\n{d['description']}\n")

    if past:
        lines.append("\n## Past Deadlines (Already in Effect)\n")
        for d in past[:5]:
            lines.append(f"- **{d['title']}** - {d['date']} ({abs(d['days_until'])} days ago)")

    return "\n".join(lines)


@mcp_tool(
    name="pki_get_frameworks",
    annotations={
        "title": "Get Regulatory Frameworks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_frameworks(params: GetFrameworksInput) -> str:
    """Get list of regulatory compliance frameworks.

    Returns frameworks like DORA, NIS2, UK CSR Bill with metadata.
    Does not include individual deadlines (use get_framework for that).

    Args:
        params: GetFrameworksInput with jurisdiction filter and response_format

    Returns:
        List of frameworks with metadata
    """
    now = datetime.now(timezone.utc)
    frameworks = get_frameworks_list(params.jurisdiction)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "frameworks": frameworks,
            "as_of": now.isoformat(),
        }, indent=2)

    lines = ["# Regulatory Compliance Frameworks\n"]
    lines.append(f"**As of:** {now.strftime('%Y-%m-%d')}\n")

    for f in frameworks:
        lines.append(f"## {f['name']} ({f['jurisdiction'].upper()})")
        lines.append(f"**{f['full_name']}**\n")
        lines.append(f"{f['description']}\n")
        if f['effective_date']:
            lines.append(f"**Effective:** {f['effective_date']}")
        lines.append(f"**Deadlines:** {f['deadline_count']} total, {f['upcoming_count']} upcoming")
        if f['next_deadline']:
            lines.append(f"**Next deadline:** {f['next_deadline']}")
        if f['resource_link']:
            lines.append(f"**Guide:** {f['resource_link']}")
        lines.append("")

    return "\n".join(lines)


@mcp_tool(
    name="pki_get_framework",
    annotations={
        "title": "Get Single Framework",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_framework(params: GetFrameworkInput) -> str:
    """Get a single regulatory framework with all its deadlines.

    Args:
        params: GetFrameworkInput with framework_id and response_format

    Returns:
        Framework details with all deadlines
    """
    now = datetime.now(timezone.utc)
    framework = get_framework_by_id(params.framework_id)

    if not framework:
        return json.dumps({"error": f"Framework '{params.framework_id}' not found"})

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "framework": framework,
            "as_of": now.isoformat(),
        }, indent=2)

    lines = [f"# {framework['name']} - {framework['full_name']}\n"]
    lines.append(f"**Jurisdiction:** {framework['jurisdiction'].upper()}")
    if framework['effective_date']:
        lines.append(f"**Effective:** {framework['effective_date']}")
    lines.append(f"\n{framework['description']}\n")
    lines.append(f"**Applies to:** {', '.join(framework['applies_to'])}\n")
    lines.append(f"**Certificate relevance:** {framework['certificate_relevance']}\n")

    if framework['resource_link']:
        lines.append(f"**Guide:** {framework['resource_link']}\n")

    lines.append("## Deadlines\n")
    for d in framework['deadlines']:
        status_icon = {"passed": "✅", "ongoing": "🔄", "upcoming": "📅"}.get(d['status'], "❓")
        lines.append(f"### {status_icon} {d['title']}")
        lines.append(f"**Date:** {d['date']} ({d['days_until']} days)")
        lines.append(f"**Status:** {d['status']}")
        lines.append(f"**Impact:** {d.get('impact', 'unknown')}")
        lines.append(f"\n{d['description']}\n")

    return "\n".join(lines)


# Set true in main() when this process IS the systemd API serving /status.
# That route calls get_status() directly, so leaving the peer fetch enabled
# there would have the API call itself back through nginx, recursively.
# Only the Docker MCP — the runtime that can actually go stale — probes.
_SERVING_PEER_API = False

PEER_API_STATUS_URL = "https://compliance-api.fixmycert.com/status"


async def _peer_api_status(unified_total: int) -> dict:
    """Compare this runtime's data_version against the live API's.

    The Docker container bakes pki_compliance_mcp.py into its image while the
    systemd API reads it from disk, so the two drift silently — as they did for
    two days from 2026-07-29, which produced a confident, wrong "the droplet is
    behind" report. deploy.sh now fails on drift, but a container started any
    other way can still be stale, so the check is worth having at read time.

    Best-effort by design: a network blip must degrade to "unknown", never
    error the tool or make a caller think it found drift.
    """
    if _SERVING_PEER_API:
        return {"peer_api_check": "skipped (this process is the API)"}

    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            r = await client.get(PEER_API_STATUS_URL, headers={"Cache-Control": "no-cache"})
            r.raise_for_status()
            peer = r.json()
    except Exception as e:
        return {
            "peer_api_check": f"unavailable: {type(e).__name__}",
            "runtimes_agree": "unknown",
        }

    peer_version = peer.get("data_version")
    peer_unified = peer.get("total_deadlines_unified")
    agree = (
        peer_version == COMPLIANCE_METADATA.get("dataVersion")
        and peer_unified == unified_total
    )

    out = {
        "peer_api_check": PEER_API_STATUS_URL,
        "api_data_version": peer_version,
        "api_total_deadlines_unified": peer_unified,
        "runtimes_agree": agree,
    }
    if not agree:
        out["drift_warning"] = (
            "This MCP and the live API disagree. This process is serving stale "
            "code — do not trust pki_get_deadlines until the container is rebuilt "
            "(run pki-compliance-mcp/deploy.sh on the droplet)."
        )
    return out


def _peer_status_line(peer: dict) -> str:
    """One-line rendering of the drift cross-check for the markdown format."""
    agree = peer.get("runtimes_agree")
    if agree is True:
        return (
            f"**Runtime cross-check:** agrees with the live API "
            f"({peer.get('api_data_version')}, "
            f"{peer.get('api_total_deadlines_unified')} unified)\n"
        )
    if agree is False:
        return (
            f"**⚠ Runtime cross-check: DRIFT.** This process is at "
            f"{COMPLIANCE_METADATA.get('dataVersion')}; the live API is at "
            f"{peer.get('api_data_version')}. {peer['drift_warning']}\n"
        )
    return f"**Runtime cross-check:** {peer.get('peer_api_check')}\n"


@mcp_tool(
    name="pki_get_status",
    annotations={
        "title": "Get Monitoring Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_status(params: GetStatusInput) -> str:
    """Get current status of the PKI compliance monitor.

    Shows last check time, tracked sources, and summary of monitoring state.

    Args:
        params: GetStatusInput with response_format

    Returns:
        Monitoring status summary
    """
    state = load_state()

    # data_version identifies the CODE this process loaded, not the data dir.
    # The Docker container bakes pki_compliance_mcp.py into its image while the
    # systemd API reads it from disk, so the two can drift apart silently — as
    # they did for two days from 2026-07-29. Exposing it here lets a caller with
    # no shell on the droplet (Cowork) compare this against the live API's
    # dataVersion and self-diagnose a stale container instead of guessing.
    unified_total = len(get_all_deadlines_unified())
    peer = await _peer_api_status(unified_total)

    result = {
        "last_check": state.get("last_check"),
        "data_version": COMPLIANCE_METADATA.get("dataVersion"),
        "data_last_updated": COMPLIANCE_METADATA.get("lastUpdated"),
        "tracked_feeds": len(FEEDS),
        "tracked_documents": len(DOCUMENTS),
        # tracked_deadlines counts DEADLINES only. It EXCLUDES the
        # REGULATORY_FRAMEWORKS entries, so it will never equal the API's
        # unified total — comparing the two is not a drift signal. Compare
        # total_deadlines_unified instead; that is the like-for-like number.
        "tracked_deadlines": len(DEADLINES),
        "total_deadlines_unified": unified_total,
        "document_hashes_stored": len(state.get("document_hashes", {})),
        # Two-runtime drift cross-check, resolved server-side. Cowork cannot
        # web_fetch the API's /status (URL-provenance rule refuses a URL read
        # out of a project file), so it could report this side's data_version
        # but never compare it. Doing the comparison here makes the whole
        # check one MCP call. See runtimes_agree for the verdict.
        **peer,
        "feeds": {k: v["name"] for k, v in FEEDS.items()},
        "documents": {k: v["name"] for k, v in DOCUMENTS.items()},
        "data_directory": str(DATA_DIR),
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    # Markdown format
    lines = [
        "# PKI Compliance Monitor Status\n",
        f"**Last check:** {state.get('last_check', 'Never')}",
        f"**Data version:** {COMPLIANCE_METADATA.get('dataVersion')} "
        f"(data last updated {COMPLIANCE_METADATA.get('lastUpdated')})",
        f"**Data directory:** `{DATA_DIR}`\n",
        _peer_status_line(peer),
        "## Tracked Sources\n",
        f"- **Feeds:** {len(FEEDS)}",
        f"- **Documents:** {len(DOCUMENTS)}",
        f"- **Deadlines:** {len(DEADLINES)} in `DEADLINES`, "
        f"{unified_total} total including regulatory frameworks\n",
        "### Feeds",
    ]

    for feed_id, feed in FEEDS.items():
        lines.append(f"- `{feed_id}`: {feed['name']} ({feed['priority']})")

    lines.append("\n### Documents")
    for doc_id, doc in DOCUMENTS.items():
        hash_status = "✓" if doc_id in state.get("document_hashes", {}) else "○"
        lines.append(f"- `{doc_id}`: {doc['name']} [{hash_status}]")

    return "\n".join(lines)

# --- Content candidates (Part A, 2026-08-07) --------------------------------
# The research pipeline's fifth outcome: real, well-sourced, no date certain —
# routed to content instead of the review queue. The ledger is written by the
# HOST cron, whose DATA_DIR is ~/.pki-compliance-mcp — and ONLY there.
#
# 2026-08-15: the ledger is resolved by an explicit path list, not by DATA_DIR
# alone. DATA_DIR is per-runtime and two of the three runtimes point somewhere
# the ledger has never been, which is what made /api/content-candidates and
# pki_list_content_candidates serve an empty ledger for a week while the file
# on disk held six rows:
#   host cron   DATA_DIR=~/.pki-compliance-mcp          <- writes the ledger
#   systemd API DATA_DIR=/opt/mcp-servers/pki-compliance-mcp/data (unit file)
#   Docker MCP  DATA_DIR=/data (volume)
# Repointing the unit's DATA_DIR would also move where state.json and news.json
# resolve for the API process, so the path is made explicit here instead and
# DATA_DIR is left alone (Pat's 2026-08-15 DECISION; mechanism was CC's call).
# Order: CONTENT_CANDIDATES_FILE env override, this process's DATA_DIR, then
# the cron dir. A readable NON-EMPTY ledger wins over a readable empty one, so
# a stray empty file cannot shadow the real one. The Docker MCP still finds
# nothing on disk (no /root/.pki-compliance-mcp in the container) and keeps
# falling back to the API route — which now answers with real rows.

CONTENT_CANDIDATES_FILE = DATA_DIR / "content_candidates.json"
CRON_DATA_DIR = Path.home() / ".pki-compliance-mcp"
PEER_API_CANDIDATES_URL = "https://compliance-api.fixmycert.com/api/content-candidates"


def _content_candidates_paths() -> List[Path]:
    """Ledger locations to try, in order, deduplicated."""
    candidates = []
    override = _os.environ.get("CONTENT_CANDIDATES_FILE")
    if override:
        candidates.append(Path(override))
    # Module constant, not DATA_DIR directly: tests redirect it.
    candidates.append(CONTENT_CANDIDATES_FILE)
    candidates.append(CRON_DATA_DIR / "content_candidates.json")
    seen, out = set(), []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _read_content_candidates() -> tuple:
    """(ledger, path) for the first readable ledger, preferring a non-empty
    one; (None, None) if no path holds a readable JSON object."""
    fallback = (None, None)
    for path in _content_candidates_paths():
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data:
            return data, path
        if fallback == (None, None):
            fallback = (data, path)
    return fallback


def _load_content_candidates_local() -> Optional[dict]:
    """The ledger, wherever it actually lives, or None if no path holds one."""
    return _read_content_candidates()[0]


@mcp_tool(
    name="pki_list_content_candidates",
    annotations={
        "title": "List Content Candidates",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def list_content_candidates(params: ListContentCandidatesInput) -> str:
    """List content candidates — pipeline items that are real and well-sourced
    but have no date certain, so they became content leads instead of tracked
    deadlines. Never part of the review queue.

    sink_status "posted" means a draft exists in the news desk; "pending" means
    delivery hasn't succeeded yet (it retries on each research run).

    Args:
        params: ListContentCandidatesInput with sink_status filter, limit,
                response_format

    Returns:
        Candidates newest-first with rule, sink status, and source URLs
    """
    ledger, ledger_path = _read_content_candidates()
    source = str(ledger_path)
    if ledger is None:
        # Container case: no ledger on any local path. Ask the systemd API,
        # which resolves the cron's ledger directly.
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                r = await client.get(PEER_API_CANDIDATES_URL, headers={"Cache-Control": "no-cache"})
                r.raise_for_status()
                ledger = r.json().get("candidates", {})
                source = PEER_API_CANDIDATES_URL
        except Exception as e:
            tried = ", ".join(str(p) for p in _content_candidates_paths())
            return (f"❌ No local ledger at any of [{tried}] and the API "
                    f"fallback failed ({type(e).__name__}: {e}). If no candidate has "
                    f"ever been classified, the ledger simply doesn't exist yet.")
    if not isinstance(ledger, dict):
        ledger = {}

    rows = []
    for sig, row in ledger.items():
        if not isinstance(row, dict):
            continue
        if params.sink_status and row.get("sink_status") != params.sink_status:
            continue
        rows.append({**row, "signature": sig})
    rows.sort(key=lambda r: (r.get("recorded_at") or r.get("first_seen") or ""), reverse=True)
    total = len(rows)
    rows = rows[:params.limit]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"candidates": rows, "total": total,
                           "shown": len(rows), "source": source}, indent=2)

    if not rows:
        flt = f" with sink_status={params.sink_status}" if params.sink_status else ""
        return f"No content candidates{flt}. (Ledger: {source})"
    lines = [f"# Content candidates ({len(rows)} of {total})",
             f"_Ledger: {source}_", ""]
    for r in rows:
        status = r.get("sink_status", "?")
        icon = {"posted": "✅", "pending": "⏳"}.get(status, "•")
        lines.append(f"- {icon} **{r.get('title', r['signature'])}** — rule `{r.get('rule', '?')}`, "
                     f"{status}, first seen {r.get('first_seen', '?')}")
        if r.get("news_id"):
            lines.append(f"  - news draft id: `{r['news_id']}`")
        primary = r.get("primary_url")
        if primary:
            lines.append(f"  - primary source: {primary}")
        for u in (r.get("provenance_urls") or []):
            if u != primary:
                lines.append(f"  - source: {u}")
    return "\n".join(lines)


@mcp_tool(
    name="pki_list_sources",
    annotations={
        "title": "List Available Sources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def list_sources(params: GetStatusInput) -> str:
    """List all available PKI compliance sources that can be monitored.

    Returns detailed information about each feed and document source,
    including URLs and priority levels.

    Args:
        params: GetStatusInput with response_format

    Returns:
        Detailed list of all sources
    """
    result = {
        "feeds": FEEDS,
        "documents": DOCUMENTS,
        "deadline_categories": list(set(d.get("category", "general") for d in DEADLINES)),
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = [
        "# PKI Compliance Sources\n",
        "## RSS/Atom Feeds\n",
        "| ID | Name | Priority | Type |",
        "|----|------|----------|------|",
    ]

    for feed_id, feed in FEEDS.items():
        lines.append(f"| `{feed_id}` | {feed['name']} | {feed['priority']} | {feed['type']} |")

    lines.append("\n## Documents\n")
    lines.append("| ID | Name | Priority |")
    lines.append("|----|------|----------|")

    for doc_id, doc in DOCUMENTS.items():
        lines.append(f"| `{doc_id}` | {doc['name']} | {doc['priority']} |")

    lines.append("\n## Deadline Categories\n")
    for cat in set(d.get("category", "general") for d in DEADLINES):
        count = sum(1 for d in DEADLINES if d.get("category") == cat)
        lines.append(f"- `{cat}`: {count} deadlines")

    return "\n".join(lines)

# ============================================================================
# Main Entry Point
# ============================================================================

# ---------------------------------------------------------------------------
# Stripe sale notifications (webhook -> Pushover "cha-ching")
# ---------------------------------------------------------------------------

STRIPE_SALES_LOG = Path("/root/.pki-compliance-mcp/stripe_sales.jsonl")


def verify_stripe_signature(payload: bytes, sig_header: str, secret: str,
                            tolerance: int = 300,
                            now: float | None = None) -> bool:
    """Verify a Stripe-Signature header (t=...,v1=...) against the payload.

    Constant-time compare over every v1 candidate; rejects timestamps
    outside `tolerance` seconds to block replay.
    """
    import hmac
    import time as _time
    ts = None
    candidates = []
    for part in (sig_header or "").split(","):
        k, _, v = part.strip().partition("=")
        if k == "t" and v.isdigit():
            ts = int(v)
        elif k == "v1" and v:
            candidates.append(v)
    if ts is None or not candidates:
        return False
    if abs((now if now is not None else _time.time()) - ts) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload,
                        hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, c) for c in candidates)


def summarize_stripe_event(event: dict) -> dict | None:
    """Return {"title", "message"} for a push-worthy event, else None."""
    if event.get("type") != "checkout.session.completed":
        return None
    obj = (event.get("data") or {}).get("object") or {}
    amount = obj.get("amount_total")
    if isinstance(amount, (int, float)):
        amount_str = f"{amount / 100:,.2f} {(obj.get('currency') or 'usd').upper()}"
    else:
        amount_str = "unknown amount"
    email = (obj.get("customer_details") or {}).get("email") or "unknown buyer"
    return {"title": "💰 FixMyCert sale", "message": f"{amount_str} — {email}"}


def send_pushover(title: str, message: str) -> bool:
    """Best-effort Pushover push with the cash-register sound."""
    token = _os.environ.get("PUSHOVER_TOKEN", "")
    user = _os.environ.get("PUSHOVER_USER", "")
    if not (token and user):
        return False
    try:
        r = httpx.post("https://api.pushover.net/1/messages.json", data={
            "token": token, "user": user, "title": title,
            "message": message, "sound": "cashregister",
        }, timeout=15)
        r.raise_for_status()
        return True
    except Exception:
        return False


# For Replit: Simple HTTP wrapper if MCP HTTP transport has issues
def create_http_app():
    """Create a simple HTTP API wrapper for environments where MCP HTTP transport doesn't work."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    import html as _html_mod
    def _esc(s: str) -> str:
        return _html_mod.escape(str(s))

    class PKIComplianceHandler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/webhooks/stripe":
                self._reply(404, b'{"error": "not found"}')
                return
            secret = _os.environ.get("STRIPE_WEBHOOK_SECRET", "")
            if not secret:
                self._reply(503, b'{"error": "webhook not configured"}')
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if not 0 < length <= 1_000_000:
                self._reply(400, b'{"error": "bad content length"}')
                return
            payload = self.rfile.read(length)
            if not verify_stripe_signature(
                    payload, self.headers.get("Stripe-Signature", ""), secret):
                self._reply(400, b'{"error": "bad signature"}')
                return
            try:
                event = json.loads(payload)
            except Exception:
                self._reply(400, b'{"error": "bad json"}')
                return
            # Always ack verified events with 200 so Stripe doesn't retry;
            # a duplicate delivery at worst repeats a cha-ching.
            sale = summarize_stripe_event(event)
            if sale:
                pushed = send_pushover(sale["title"], sale["message"])
                try:
                    STRIPE_SALES_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with open(STRIPE_SALES_LOG, "a") as f:
                        f.write(json.dumps({
                            "at": datetime.now(timezone.utc).isoformat(),
                            "event_id": event.get("id"),
                            "summary": sale["message"],
                            "pushed": pushed,
                        }) + "\n")
                except Exception:
                    pass
            self._reply(200, b'{"received": true}')

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            # Health check
            if path == "/" or path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "service": "pki-compliance-monitor",
                    "endpoints": [
                        "/health", "/status", "/feeds", "/documents", "/sources",
                        "/deadlines", "/frameworks", "/frameworks/{id}",
                        "/api/compliance/deadlines", "/api/compliance/upcoming",
                        "/api/compliance/frameworks", "/api/compliance/frameworks/{id}",
                        "/api/compliance/deadlines.csv",
                        "/api/compliance-data", "/api/news", "/api/news/sources", "/api/news/refresh",
                        "/api/content-candidates"
                    ]
                }).encode())
                return
            
            # Dashboard — read-only HTML view, token-protected
            if path == "/dashboard":
                token = params.get("token", [None])[0]
                expected = _os.environ.get("DASHBOARD_TOKEN", "")
                if not expected or token != expected:
                    self.send_response(401)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Unauthorized. Append ?token=YOUR_TOKEN to the URL.")
                    return

                now = datetime.now(timezone.utc)
                unified = get_all_deadlines_unified()
                upcoming = sorted(
                    [d for d in unified if days_until(d["date"]) >= 0],
                    key=lambda d: d["date"],
                )[:20]
                news_data = get_news(limit=15)
                news_items = news_data.get("items", [])

                # Check for pending auto-refresh updates
                pending_files = sorted(DATA_DIR.glob("pending_updates_*.json"), reverse=True)
                pending_html = ""
                if pending_files:
                    latest = pending_files[0]
                    try:
                        pending = json.loads(latest.read_text())
                        pdate = latest.stem.replace("pending_updates_", "")
                        items = []
                        for d in pending.get("new_deadlines", []):
                            items.append(f'<li class="new">NEW: {_esc(d.get("title",""))} ({d.get("date","")})</li>')
                        for d in pending.get("document_version_updates", []):
                            items.append(f'<li class="update">DOC: {_esc(d.get("id",""))} &rarr; {_esc(d.get("new_version",""))}</li>')
                        for d in pending.get("updated_deadlines", []):
                            items.append(f'<li class="update">UPDATED: {_esc(d.get("id",""))}</li>')
                        for d in pending.get("needs_human_review", []):
                            reason = d.get("reason", d) if isinstance(d, dict) else str(d)
                            items.append(f'<li class="review">REVIEW: {_esc(str(reason))}</li>')
                        summary = _esc(pending.get("summary", ""))
                        if items:
                            pending_html = f'''
                            <section>
                                <h2>Pending Updates ({pdate})</h2>
                                <p class="summary">{summary}</p>
                                <ul>{"".join(items)}</ul>
                            </section>'''
                        else:
                            pending_html = f'<section><h2>Pending Updates ({pdate})</h2><p>No changes found.</p></section>'
                    except Exception:
                        pending_html = ""

                # Build deadline rows
                deadline_rows = []
                for d in upcoming:
                    days_left = days_until(d["date"])
                    urgency = "urgent" if days_left < 90 else ("soon" if days_left < 365 else "")
                    source = d.get("source", d.get("framework_id", ""))
                    deadline_rows.append(
                        f'<tr class="{urgency}">'
                        f'<td>{d["date"]}</td>'
                        f'<td>{days_left}d</td>'
                        f'<td>{_esc(d.get("title",""))}</td>'
                        f'<td>{_esc(source)}</td>'
                        f'<td>{"Yes" if d.get("isMajor") else ""}</td>'
                        f'</tr>'
                    )

                # Build news rows
                news_rows = []
                for item in news_items:
                    pub = item.get("publishedAt", "")[:10]
                    prio = ' class="priority"' if item.get("isPriority") else ""
                    news_rows.append(
                        f'<tr{prio}>'
                        f'<td>{pub}</td>'
                        f'<td><a href="{_esc(item.get("url",""))}" target="_blank" rel="noopener">{_esc(item.get("title",""))}</a></td>'
                        f'<td>{_esc(item.get("source",""))}</td>'
                        f'</tr>'
                    )

                stale = is_data_stale()
                stale_banner = '<div class="stale-banner">Data is stale — last review was over 45 days ago</div>' if stale else ""

                html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>PKI Compliance Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:1.5rem;max-width:1100px;margin:0 auto}}
h1{{color:#38bdf8;margin-bottom:.5rem;font-size:1.5rem}}
h2{{color:#94a3b8;margin:1.5rem 0 .75rem;font-size:1.1rem;text-transform:uppercase;letter-spacing:.05em}}
.meta{{color:#64748b;font-size:.85rem;margin-bottom:1rem}}
.stale-banner{{background:#7c2d12;color:#fed7aa;padding:.5rem 1rem;border-radius:6px;margin-bottom:1rem;font-weight:600}}
section{{background:#1e293b;border-radius:8px;padding:1rem 1.25rem;margin-bottom:1rem}}
table{{width:100%;border-collapse:collapse;font-size:.875rem}}
th{{text-align:left;color:#94a3b8;padding:.5rem .4rem;border-bottom:1px solid #334155}}
td{{padding:.45rem .4rem;border-bottom:1px solid #1e293b;vertical-align:top}}
tr.urgent td{{color:#fbbf24;font-weight:600}}
tr.soon td{{color:#a5b4fc}}
tr.priority td{{background:#1e1b4b}}
a{{color:#38bdf8;text-decoration:none}}
a:hover{{text-decoration:underline}}
ul{{list-style:none;padding:0}}
li{{padding:.3rem 0;padding-left:1rem;position:relative}}
li::before{{content:"";position:absolute;left:0;top:.65rem;width:6px;height:6px;border-radius:50%}}
li.new::before{{background:#34d399}}
li.update::before{{background:#60a5fa}}
li.review::before{{background:#fbbf24}}
.summary{{color:#94a3b8;font-size:.85rem;margin-bottom:.5rem}}
</style>
</head>
<body>
<h1>PKI Compliance Dashboard</h1>
<p class="meta">Generated {now.strftime("%Y-%m-%d %H:%M UTC")} &middot; Data version {COMPLIANCE_METADATA.get("dataVersion","")} &middot; Last review {DATA_FRESHNESS.get("lastFullReview","")}</p>
{stale_banner}
{pending_html}
<section>
<h2>Upcoming Deadlines (next 20)</h2>
<table>
<tr><th>Date</th><th>In</th><th>Deadline</th><th>Source</th><th>Major</th></tr>
{"".join(deadline_rows)}
</table>
</section>
<section>
<h2>PKI News Feed</h2>
<table>
<tr><th>Date</th><th>Article</th><th>Source</th></tr>
{"".join(news_rows) if news_rows else "<tr><td colspan='3'>No news items yet. <a href='/api/news/refresh?token=" + (token or "") + "'>Refresh feeds</a></td></tr>"}
</table>
</section>
<section>
<h2>Document Versions</h2>
<table>
<tr><th>Document</th><th>Version</th><th>Date</th></tr>
{"".join(f'<tr><td>{_esc(d["name"])}</td><td>{_esc(d["version"])}</td><td>{_esc(d["date"])}</td></tr>' for d in CABF_DOCUMENTS)}
</table>
</section>
</body>
</html>'''

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Robots-Tag", "noindex")
                self.end_headers()
                self.wfile.write(html.encode())
                return

            # News API endpoints
            if path == "/api/news":
                category = params.get("category", [None])[0]
                limit = min(int(params.get("limit", [20])[0]), 50)
                offset = int(params.get("offset", [0])[0])
                
                result = get_news(category=category, limit=limit, offset=offset)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=1800")
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
                return
            
            if path == "/api/news/sources":
                result = get_news_sources()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
                return
            
            if path == "/api/news/refresh":
                result = fetch_all_news_feeds()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "refreshed",
                    "feedsChecked": result.get("feedsChecked", 0),
                    "newItemsAdded": result.get("newItemsAdded", 0),
                    "totalItems": len(result.get("items", [])),
                    "updatedAt": result.get("lastFetched")
                }, indent=2).encode())
                return

            # Content-candidate ledger (Part A, 2026-08-07). Served by the
            # systemd API, which resolves the cron's ledger by explicit path
            # (see _content_candidates_paths — its own DATA_DIR points at a
            # directory that has never held the file). The Docker MCP's
            # pki_list_content_candidates falls back to this route because
            # its /data volume never contains the ledger either.
            # Read-only, no cache: sink_status changes on every research run.
            if path == "/api/content-candidates":
                ledger, ledger_path = _read_content_candidates()
                ledger = ledger or {}
                pending_count = sum(1 for r in ledger.values()
                                    if isinstance(r, dict) and r.get("sink_status") == "pending")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "candidates": ledger,
                    "total": len(ledger),
                    "pending_delivery": pending_count,
                    # Which path answered — so an empty ledger can be told
                    # apart from a ledger read out of the wrong directory.
                    "source": str(ledger_path) if ledger_path else None,
                }, indent=2).encode())
                return

            # NEW: All compliance data in one call (for frontend)
            if path == "/api/compliance-data":
                now = datetime.now(timezone.utc)

                # Get unified deadlines with framework_id, is_estimated, etc.
                unified_deadlines = get_all_deadlines_unified()
                deadlines_with_countdown = []
                for d in unified_deadlines:
                    # days_until, not a second inline copy of the arithmetic:
                    # this route carried its own timestamp-difference version
                    # and inherited the same floor-toward-past off-by-one.
                    days = days_until(d["date"])
                    deadlines_with_countdown.append({
                        **d,
                        "daysUntil": days,
                        "isPast": days < 0,
                    })

                response_data = {
                    "deadlines": deadlines_with_countdown,
                    "cabfDocuments": CABF_DOCUMENTS,
                    "rootStores": ROOT_STORES,
                    "rootStoreComparison": ROOT_STORE_COMPARISON,
                    "caAcquisitions": CA_ACQUISITIONS,
                    "caChains": CA_CHAINS,
                    "caChainQuickReference": CA_CHAIN_QUICK_REFERENCE,
                    "relatedRfcs": RELATED_RFCS,
                    "metadata": {
                        **COMPLIANCE_METADATA,
                        "fetchedAt": now.isoformat(),
                    },
                    "dataFreshness": DATA_FRESHNESS,
                    "staleWarning": is_data_stale(),
                    "nist800131a": NIST_800_131A,
                    "pciDss": PCI_DSS_V4,
                    "fips140": FIPS_140,
                    "nist80057": NIST_800_57,
                    "nist80052": NIST_800_52,
                    "nistPqc": NIST_PQC,
                    "frameworkCrossReference": FRAMEWORK_CROSS_REFERENCE,
                    "regulatoryFrameworks": get_frameworks_list(),
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=3600")  # Cache for 1 hour
                self.end_headers()
                self.wfile.write(json.dumps(response_data, indent=2).encode())
                return

            # CSV export of deadlines. Same filters as /api/compliance/deadlines
            # (category, framework, jurisdiction, status, within_days); a plain
            # <a href download> in the frontend hits this. utf-8-sig BOM so Excel
            # renders accented / em-dash characters.
            if path == "/api/compliance/deadlines.csv":
                within_days = params.get("within_days", [None])[0]
                csv_text = build_deadlines_csv(
                    category=params.get("category", [None])[0],
                    framework=params.get("framework", [None])[0],
                    jurisdiction=params.get("jurisdiction", [None])[0],
                    status=params.get("status", [None])[0],
                    within_days=int(within_days) if within_days else None,
                )
                filename = f"pki-compliance-deadlines-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(csv_text.encode("utf-8-sig"))
                return

            # Run async handlers synchronously
            import asyncio

            try:
                if path == "/status":
                    result = asyncio.run(get_status(GetStatusInput(
                        response_format=ResponseFormat.JSON
                    )))
                elif path == "/feeds":
                    feed_ids = params.get("feed_ids", [None])[0]
                    feed_list = feed_ids.split(",") if feed_ids else None
                    result = asyncio.run(check_feeds(CheckFeedsInput(
                        feed_ids=feed_list,
                        since_days=int(params.get("since_days", [7])[0]),
                        response_format=ResponseFormat.JSON
                    )))
                elif path == "/documents":
                    doc_id = params.get("document_id", [None])[0]
                    if doc_id:
                        result = asyncio.run(check_document(CheckDocumentInput(
                            document_id=doc_id,
                            response_format=ResponseFormat.JSON
                        )))
                    else:
                        result = asyncio.run(check_all_documents(CheckAllDocumentsInput(
                            priority=params.get("priority", [None])[0],
                            response_format=ResponseFormat.JSON
                        )))
                elif path == "/deadlines" or path == "/api/compliance/deadlines":
                    within_days = params.get("within_days", [None])[0]
                    result = asyncio.run(get_deadlines(GetDeadlinesInput(
                        category=params.get("category", [None])[0],
                        within_days=int(within_days) if within_days else None,
                        framework=params.get("framework", [None])[0],
                        jurisdiction=params.get("jurisdiction", [None])[0],
                        status=params.get("status", [None])[0],
                        response_format=ResponseFormat.JSON
                    )))
                elif path == "/api/compliance/upcoming":
                    within_days = params.get("days", [90])[0]
                    result = asyncio.run(get_deadlines(GetDeadlinesInput(
                        within_days=int(within_days),
                        jurisdiction=params.get("jurisdiction", [None])[0],
                        status="upcoming",
                        response_format=ResponseFormat.JSON
                    )))
                elif path == "/frameworks" or path == "/api/compliance/frameworks":
                    result = asyncio.run(get_frameworks(GetFrameworksInput(
                        jurisdiction=params.get("jurisdiction", [None])[0],
                        response_format=ResponseFormat.JSON
                    )))
                elif path.startswith("/frameworks/") or path.startswith("/api/compliance/frameworks/"):
                    if path.startswith("/api/compliance/frameworks/"):
                        framework_id = path.split("/api/compliance/frameworks/")[1]
                    else:
                        framework_id = path.split("/frameworks/")[1]
                    result = asyncio.run(get_framework(GetFrameworkInput(
                        framework_id=framework_id,
                        response_format=ResponseFormat.JSON
                    )))
                elif path == "/sources":
                    result = asyncio.run(list_sources(GetStatusInput(
                        response_format=ResponseFormat.JSON
                    )))
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not found"}).encode())
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                # Result is already JSON string from our tools
                self.wfile.write(result.encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        def log_message(self, format, *args):
            print(f"[{datetime.now().isoformat()}] {args[0]}")

    return PKIComplianceHandler


if __name__ == "__main__":
    import sys
    import os

    # Parse command line args
    use_http = "--http" in sys.argv
    use_simple_http = "--simple-http" in sys.argv or os.environ.get("REPLIT_DEPLOYMENT")

    # MCP_TRANSPORT env var (used by Docker Compose, consistent with other MCP servers)
    mcp_transport = os.environ.get("MCP_TRANSPORT", "")

    # Replit provides PORT env var
    port = int(os.environ.get("PORT", os.environ.get("MCP_PORT", 5000)))

    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    if mcp_transport == "sse":
        # Docker Compose MCP mode (consistent with other servers)
        if not MCP_AVAILABLE:
            print("Error: MCP SDK not installed. Install with: pip install 'mcp[cli]'")
            sys.exit(1)
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = port
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        print(f"PKI Compliance MCP server (SSE) on http://0.0.0.0:{port}")
        mcp.run(transport="sse")
    elif use_simple_http or os.environ.get("REPLIT_DEPLOYMENT"):
        # This process IS the API behind compliance-api.fixmycert.com, so its
        # /status must not fetch that URL — it would call itself through nginx.
        globals()["_SERVING_PEER_API"] = True
        # Simple HTTP server for Replit / systemd API mode.
        # ThreadingHTTPServer (not HTTPServer) — single-threaded server hangs on
        # any slow/stuck handler, taking out the whole API. Public endpoint sees
        # malicious probes (phpunit, PROPFIND) that can wedge a single-threaded
        # listener.
        from http.server import ThreadingHTTPServer as HTTPServer
        handler = create_http_app()
        server = HTTPServer(("0.0.0.0", port), handler)
        print(f"PKI Compliance Monitor running on http://0.0.0.0:{port}")
        print(f"   Endpoints: /health, /status, /feeds, /documents, /deadlines, /frameworks, /sources")
        server.serve_forever()
    elif use_http:
        if not MCP_AVAILABLE:
            print("Error: MCP SDK not installed. Use --simple-http or install with: pip install mcp")
            sys.exit(1)
        print(f"Starting PKI Compliance MCP server on http://0.0.0.0:{port}")
        mcp.run(transport="streamable_http", host="0.0.0.0", port=port)
    else:
        if not MCP_AVAILABLE:
            print("Error: MCP SDK not installed. Use --simple-http or install with: pip install mcp")
            sys.exit(1)
        mcp.run()  # stdio transport (default)