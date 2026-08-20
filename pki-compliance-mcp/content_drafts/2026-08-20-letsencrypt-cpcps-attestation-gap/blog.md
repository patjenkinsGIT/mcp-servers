# Let's Encrypt's CP/CPS Is Missing a Required Attestation — What Your Cert Team Needs to Know

On August 10, 2026, Let's Encrypt disclosed on its community forum that its Certificate Policy/Certification Practice Statement (CP/CPS) does not contain the attestation of compliance with the Chrome Root Program Policy and the CCADB Policy that was required by June 15, 2026. Because Let's Encrypt is one of the largest public CAs on the web — and the backbone of most ACME automation — any open question about whether this gap could trigger revocation deserves your attention this week, even though a mandated revocation has **not** been confirmed as of this writing.

## What happened

Root programs (Chrome, and by extension CCADB) require CAs to publish an explicit attestation of policy compliance in their CP/CPS by a set effective date. Let's Encrypt became aware that its CP/CPS never contained that attestation by the June 15, 2026 deadline. The gap surfaced publicly in a Let's Encrypt community thread on August 10, with a follow-up discussion thread and a corresponding Mozilla Bugzilla bug (#2038351) opened to track it. As of August 10–12, 2026, the community and CA/Browser ecosystem are actively debating whether a missing attestation is a documentation oversight or a policy violation serious enough to trigger mandated certificate revocation under existing root program rules. No resolution, remediation plan, or revocation mandate has been published yet.

## Who is affected

- **Anyone with certificates issued by Let's Encrypt** — check your issuer field for ISRG Root X1/X2, or intermediates R3, R10, R11, E5, E6. This includes manual issuance and, more commonly, ACME automation (Certbot, acme.sh, cert-manager, and similar tooling).
- **Enterprise teams whose PKI is entirely from other CAs** (DigiCert, Sectigo, internal/private PKI) generally need no action from this specific event — but confirm you don't have Let's Encrypt certs hiding in dev, staging, or shadow-IT environments, which is extremely common.
- **CAs and root program participants** are the ones actually resolving the compliance question. If you don't operate a CA, this is a monitoring item, not (yet) a fire drill.

## Key dates

- **2026-06-15** — Effective date by which Chrome Root Program-trusted CAs, including Let's Encrypt, were required to have a CP/CPS attestation of compliance with the Chrome Root Program Policy and CCADB Policy.
- **2026-08-10** — Let's Encrypt publishes community disclosure acknowledging the CP/CPS gap.
- **2026-08-12** — Continued community and ecosystem discussion (including Mozilla Bugzilla #2038351) on whether the gap constitutes a violation requiring mandated revocation.
- **TBD** — No confirmed remediation timeline or revocation mandate exists yet. Treat this as unresolved until an official ruling is published.

## What to do now

1. **Inventory every Let's Encrypt-issued certificate** in your environment. Pull issuer data across all environments, including subdomains, internal tools, and anything managed by developers outside central cert tooling.
2. **Map your automation dependencies.** Identify which ACME clients, cron jobs, and cert-manager clusters rely on Let's Encrypt, and confirm how quickly you could cut over to another CA if issuance or trust were disrupted.
3. **Do not preemptively revoke or rotate certificates.** There is no confirmed mandate yet. Acting early wastes effort and risks breaking automation for no confirmed reason.
4. **Assign an owner to monitor the official threads** — the Let's Encrypt community posts and the Mozilla Bugzilla bug — for the actual resolution or any CCADB/Chrome ruling.
5. **Pressure-test your multi-CA contingency plan now**, while there's no deadline urgency. Prior Let's Encrypt incidents have carried revocation windows as short as a few days — you want that plan ready before you need it, not during.

This is a fast-moving, unresolved situation. We're tracking it alongside every other active PKI compliance deadline at [fixmycert.com/compliance](https://fixmycert.com/compliance) — check back there for updates as the Let's Encrypt and root program community reach a decision.
