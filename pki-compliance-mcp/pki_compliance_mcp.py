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
        "url": "https://lists.cabforum.org/pipermail/public/",
        "type": "mailman_archive",
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
    "nist_800_131a": {
        "name": "NIST SP 800-131A (Algorithm Transitions)",
        "url": "https://csrc.nist.gov/publications/detail/sp/800-131a/rev-2/final",
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
DEADLINES = [
    {
        "id": "chrome-digicert-legacy-roots-distrust",
        "date": "2026-07-01",
        "title": "Chrome distrust: DigiCert Trusted Root G4 / Assured ID G2/G3 (new issuance)",
        "description": "Chrome Root Program removes DigiCert Trusted Root G4, Assured ID G2, and Assured ID G3 (not dedicated TLS hierarchies — they also issued Code Signing/Timestamp ICAs). TLS certificates issued on or after 2026-07-01 from these roots are not Chrome-trusted; certificates issued before remain trusted until expiry. New issuance and reissues must chain to DigiCert Global Root G2 (RSA) or G3 (ECC).",
        "source": "chrome",
        "source_url": "https://knowledge.digicert.com/alerts/google-chrome-root-removal-trusted-root-g4-assured-id-g2-id-g3",
        "category": "root-store",
        "isMajor": True,
        "impact": "DigiCert customers on legacy roots must ensure new and reissued TLS certificates chain to Global Root G2 (RSA) or G3 (ECC) to remain Chrome-trusted.",
        "is_estimated": False,
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
    },
    {
        "id": "chrome-clientauth-ica",
        "date": "2025-06-15",
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
        "impact": "CAs without documented procedures may face removal from Mozilla root store."
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
        "title": "Chrome Entrust Distrust",
        "description": "Chrome distrusts TLS certificates issued by Entrust after this date.",
        "source": "chrome",
        "source_url": "https://security.googleblog.com/2024/06/sustaining-digital-certificate-security.html",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will show security warnings in Chrome."
    },
    {
        "id": "apple-entrust-distrust",
        "date": "2024-11-15",
        "title": "Apple Entrust Distrust",
        "description": "Apple distrusts TLS certificates issued by Entrust after this date.",
        "source": "apple",
        "source_url": "https://support.apple.com/en-us/121668",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will not be trusted on macOS/iOS."
    },
    {
        "id": "mozilla-entrust-distrust",
        "date": "2024-11-30",
        "title": "Mozilla Entrust Distrust",
        "description": "Firefox distrusts TLS certificates issued by Entrust after this date.",
        "source": "mozilla",
        "source_url": "https://groups.google.com/a/mozilla.org/g/dev-security-policy/c/jCvkhBjg9Yw",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will show security warnings in Firefox."
    },
    {
        "id": "microsoft-entrust-distrust",
        "date": "2025-04-16",
        "title": "Microsoft Entrust Distrust",
        "description": "Microsoft distrusts TLS certificates issued by Entrust after this date.",
        "source": "microsoft",
        "source_url": "https://learn.microsoft.com/en-us/security/trusted-root/2025/february-2025",
        "category": "certificates",
        "isMajor": True,
        "impact": "Certificates from Entrust CAs will not be trusted in Windows/Edge."
    },
    # 2026-07-21 blind-window review: entries below recovered from the ~9 months
    # the Microsoft source pointed at a dead learn.microsoft.com page.
    {
        "id": "microsoft-april-2026-ctl-notbefore",
        "date": "2026-05-19",
        "title": "Microsoft NotBefore distrust: SwissSign Silver G2, SecureSign RootCA11/CA12, ANCERT (new issuance)",
        "description": "April 2026 CTL release (published Apr 28, 2026). Certificates issued after May 19, 2026 are no longer trusted on Windows for: SwissSign Silver CA - G2, Cybertrust Japan SecureSign RootCA11 and CA12, two ANCERT roots, and Byte Root CA (full distrust); plus S/MIME-only NotBefore for 19 roots including legacy GoDaddy Class 2/gdroot-g2/sfroot-g2, HARICA 2015 RSA/ECC, CFCA EV, Camerfirma, and Firmaprofesional. Camerfirma CommerceRoot and Nets DanID OCES were disabled outright. Certificates issued before the date remain trusted.",
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
        "title": "Microsoft TRP: single-purpose roots + 10-year max validity for new root submissions",
        "description": "Trusted Root Program Requirements v1.2 (May 20, 2026), effective for root certificates submitted on or after July 1, 2026: TLS server authentication, S/MIME, and code signing must be separate dedicated trust anchors (only Client Authentication may be combined, plus Time Stamping on code-signing roots), and newly minted roots are capped at 10 years validity from submission. Multi-EKU roots submitted before January 1, 2027 remain trusted unless Microsoft directs otherwise. v1.2 also mandates public incident disclosure in Bugzilla per the CCADB incident-report format.",
        "source": "microsoft",
        "source_url": "https://github.com/TrustedRootProgram/Program-Requirements/blob/main/Requirements.md",
        "category": "root-store",
        "isMajor": False,
        "impact": "No action unless you operate a CA. Roots submitted on or after 2026-07-01 must be single-purpose and capped at 10 years; multi-EKU roots submitted before 2027-01-01 stay trusted, so CAs with multi-purpose hierarchies have until then to split them.",
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
        "description": "Per NIST SP 800-131A Rev 3, SHA-1, SHA-224, and AES-ECB mode are deprecated. Minimum security strength increases from 112 bits to 128 bits.",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/sp/800/131/a/r3/ipd",
        "category": "pqc",
        "isMajor": True,
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
        "description": "Under NIST IR 8547 timeline, quantum-vulnerable algorithms (RSA, ECDSA, DH, ECDH) will be removed from NIST standards.",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/ir/8547/ipd",
        "category": "pqc",
        "isMajor": True,
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
        "description": "End of approved use of 112-bit security strength for federal applications. Affects RSA-2048, ECC P-224, and 3-key 3DES. Drives minimum to 128-bit equivalent (RSA-3072 / ECC P-256 or stronger).",
        "source": "nist",
        "source_url": "https://csrc.nist.gov/pubs/sp/800/131/a/r3/ipd",
        "category": "algorithm-deprecation",
        "isMajor": True,
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
        "id": "digicert-g1-root-distrust",
        "date": "2026-04-15",
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
                "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554",
                "category": "effective",
                "impact": "All EU financial entities must comply with DORA's ICT risk management, incident reporting, and third-party risk requirements - no transitional period.",
                "isMajor": True,
                "description": "Regulation now applies to all EU financial entities. No transitional period."
            },
            {
                "id": "dora-roi-submission",
                "title": "ROI Submission to ESAs",
                "date": "2025-04-30",
                "source_url": "https://www.eba.europa.eu/publications-and-media/press-releases/esas-provide-roadmap-towards-designation-ctpps-under-dora",
                "category": "reporting",
                "impact": "Financial entities must have their Registers of Information on ICT third-party arrangements complete and submitted through national competent authorities.",
                "isMajor": True,
                "description": "National competent authorities submit Registers of Information on ICT third-party arrangements to ESAs."
            },
            {
                "id": "dora-ctpp-designation",
                "title": "CTPP Designation Notifications",
                "date": "2025-07-31",
                "source_url": "https://www.eba.europa.eu/publications-and-media/press-releases/esas-provide-roadmap-towards-designation-ctpps-under-dora",
                "category": "oversight",
                "impact": "Designated CTPPs come under direct ESA oversight; financial entities should verify which of their ICT providers are designated.",
                "isMajor": True,
                "description": "ESAs notify ICT third-party service providers of their classification as Critical Third-Party Providers."
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
                "description": "Second annual Register of Information submission. From 2026 onwards, competent authorities submit RoIs to the ESAs by 31 March each year (per EBA DORA reporting FAQ). Regulators expect more mature submissions with detailed ICT third-party documentation."
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
                "jurisdiction_detail": "EU"
            },
            {
                "id": "nis2-germany-bsi",
                "title": "Germany BSI Act Effective",
                "date": "2025-12-06",
                "source_url": "https://www.recht.bund.de/bgbl/1/2025/301/VO.html",
                "category": "national",
                "impact": "Entities in scope in Germany become subject to the amended BSI Act's registration, security, and reporting obligations.",
                "isMajor": True,
                "description": "German NIS2 implementation via amended BSI Act enters into force.",
                "jurisdiction_detail": "Germany"
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
                "jurisdiction_detail": "Germany"
            },
            {
                "id": "nis2-italy-audit",
                "title": "Italy First Compliance Audit",
                "date": "2026-06-30",
                "source_url": "https://www.acn.gov.it/portale/nis/categorizzazione",
                "category": "audit",
                "impact": "Italian in-scope entities must be able to demonstrate NIS2 compliance in the first audit cycle.",
                "isMajor": True,
                "description": "Deadline for first audit verifying NIS2 compliance (moved from Dec 31, 2025).",
                "jurisdiction_detail": "Italy"
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
                "jurisdiction_detail": "Luxembourg"
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
                "jurisdiction_detail": "Austria"
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
                "is_estimated": True
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
                "description": "Bill passed the House of Commons (third reading 16 Jun 2026) and had its first reading in the House of Lords on 17 June 2026. Lords second reading scheduled 14 July 2026."
            },
            {
                "id": "uk-csr-royal-assent",
                "title": "Royal Assent (Estimated)",
                "date": "2026-12-31",
                "source_url": "https://bills.parliament.uk/bills/4035",
                "category": "legislative",
                "impact": "Once the bill becomes law, secondary legislation and enforcement timelines start; in-scope organizations should begin gap assessments.",
                "isMajor": True,
                "description": "Bill was carried over and reintroduced May 2026; passed Commons third reading 16 Jun 2026; Lords second reading scheduled 14 Jul 2026. Royal Assent now expected late 2026.",
                "is_estimated": True
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
        "version": "2.2.8",
        "date": "Jun 2026",
        "url": "https://cabforum.org/working-groups/server/baseline-requirements/documents/",
    },
    {
        "id": "ev-guidelines",
        "name": "EV Guidelines",
        "version": "2.0.2",
        "date": "May 2026",
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
        "version": "1.0.14",
        "date": "May 2026",
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
        "version": "2024",
        "url": "https://github.com/apple/apple-root-program",
        "platforms": ["Safari", "iOS", "iPadOS", "macOS"],
        "keyRequirements": [
            "Certificate Transparency required",
            "OCSP checking enabled",
            "ARI implementation disclosure by April 2025",
            "Initiated 398-day max validity (2020)"
        ],
        "recentActions": [
            {"action": "Entrust distrust", "date": "2024-11-15", "description": "Safari/iOS distrusts Entrust for TLS, S/MIME, timestamping"}
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
    "lastUpdated": "2026-07-21",
    "dataVersion": "2.4.7",
    "basedOn": "CA/B Forum TLS BR 2.2.8, Code Signing BR 3.11, EV Guidelines 2.0.2, S/MIME BR 1.0.14, SC-080/081/085/090/091/092/097/098/099 Ballots, Chrome Root Program v1.8, Mozilla Root Store Policy v3.1, Apple Root Store Policy, Microsoft Trusted Root Program Requirements v1.2, NIST SP 800-131A Rev 3, NIST SP 800-57 Rev 5, NIST FIPS 203/204/205 (PQC), NIST IR 8547, NSA CNSA 2.0, PCI DSS v4.0.1, DORA (EU), NIS2 (EU), UK CSR Bill",
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
    "lastFullReview": "2026-07-21",
    "nextReviewDue": "2026-08-20",
    "reviewIntervalDays": 30,
    "fieldVerifications": {
        "deadlines": {"verified": "2026-07-21", "source": "2026-07-21 Microsoft TRP blind-window review (Oct 2025-Jul 2026): added microsoft-april-2026-ctl-notbefore (NotBefore 2026-05-19, verified against trusted-root/2026/april-2026.md), microsoft-trp-single-purpose-roots (2026-07-01, Requirements.md v1.2 sections 3.1.8/3.4.3), microsoft-pqc-tls-pilot (2026-06-08, PQC Pilot Program.md V1.0). SC0101v2 (IPR ~Aug 6) and SMC017v2 (IPR ~Jul 30) remain held. Prior review 2026-07-13"},
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
                    "nist_800_131a, nist_800_57"
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


def _strip_balanced_blocks(content: str, tag: str, start_pat: "re.Pattern") -> str:
    """Remove every region from a start_pat match through its balanced </tag>.

    If the markup is unbalanced (no matching close tag), only the opening tag
    is dropped so the scan always terminates.
    """
    tag_pat = re.compile(rf"<(/?){tag}\b[^>]*>", re.IGNORECASE)
    while True:
        m = start_pat.search(content)
        if not m:
            return content
        depth, cut_end = 1, m.end()
        for t in tag_pat.finditer(content, m.end()):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                cut_end = t.end()
                break
        content = content[:m.start()] + content[cut_end:]


def hash_content(content: str) -> str:
    """Generate SHA-256 hash of content, ignoring per-request dynamic noise."""
    # Scripts first: script bodies may contain literal tag text that would
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
    """Calculate days until a date."""
    target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (target - now).days


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
    Otherwise, calculates based on whether date is in the past.
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

    result = {
        "last_check": state.get("last_check"),
        "tracked_feeds": len(FEEDS),
        "tracked_documents": len(DOCUMENTS),
        "tracked_deadlines": len(DEADLINES),
        "document_hashes_stored": len(state.get("document_hashes", {})),
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
        f"**Data directory:** `{DATA_DIR}`\n",
        "## Tracked Sources\n",
        f"- **Feeds:** {len(FEEDS)}",
        f"- **Documents:** {len(DOCUMENTS)}",
        f"- **Deadlines:** {len(DEADLINES)}\n",
        "### Feeds",
    ]

    for feed_id, feed in FEEDS.items():
        lines.append(f"- `{feed_id}`: {feed['name']} ({feed['priority']})")

    lines.append("\n### Documents")
    for doc_id, doc in DOCUMENTS.items():
        hash_status = "✓" if doc_id in state.get("document_hashes", {}) else "○"
        lines.append(f"- `{doc_id}`: {doc['name']} [{hash_status}]")

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
                        "/api/compliance-data", "/api/news", "/api/news/sources", "/api/news/refresh"
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

            # NEW: All compliance data in one call (for frontend)
            if path == "/api/compliance-data":
                now = datetime.now(timezone.utc)

                # Get unified deadlines with framework_id, is_estimated, etc.
                unified_deadlines = get_all_deadlines_unified()
                deadlines_with_countdown = []
                for d in unified_deadlines:
                    target = datetime.strptime(d["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    days = (target - now).days
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