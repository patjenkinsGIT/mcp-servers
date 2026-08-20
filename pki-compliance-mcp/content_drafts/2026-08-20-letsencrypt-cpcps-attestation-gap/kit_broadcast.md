<!-- OPERATOR NOTES - do not send as-is:
     audience: FixMyCert brand ONLY (brand:fmc, tag 19302998) - NOT the full list
     sender:   Patrick from FixMyCert <patrick@fixmycert.com> (account default)
     This is a DRAFT. Create in Kit as a draft broadcast; never auto-send. -->

## Subject

Let's Encrypt CP/CPS is missing a required attestation

## Preview text

Community debating mandated revocation since Aug 10 — check your LE certs first

## Body

Hey — quick one, and it's genuinely unresolved as I write this.

**What happened:** Let's Encrypt disclosed on August 10 that its CP/CPS never got the attestation of compliance with the Chrome Root Program Policy and CCADB Policy that was due by June 15. The community forum and a Mozilla Bugzilla thread are now debating whether this gap is enough to trigger mandated revocation of Let's Encrypt-issued certificates. No ruling yet.

**Who's affected:** Anyone running certificates issued by Let's Encrypt — check for ISRG Root X1/X2 or intermediates R3, R10, R11 in your chain. If your whole inventory is DigiCert, Sectigo, or private PKI, you can mostly sit this one out — but double-check dev/staging environments, because LE certs hide there more often than people expect.

**Key dates:**
- June 15, 2026 — attestation deadline that was missed
- Aug 10, 2026 — Let's Encrypt discloses the gap
- Aug 12, 2026 — ecosystem debate ongoing, no resolution yet
- TBD — no confirmed remediation or revocation mandate published

**What I'd do this week:**
1. Pull a full inventory of every cert issued by Let's Encrypt across every environment, not just production.
2. Map which ACME clients and renewal jobs depend on LE, so you know your blast radius.
3. Don't rotate or revoke anything preemptively — there's no confirmed mandate, and jumping the gun just breaks automation.
4. Put someone on watch duty for the official threads and the Bugzilla bug, and have your multi-CA fallback ready in case a short revocation window gets announced.

I'll update the tracker the moment there's an actual ruling either way.

https://fixmycert.com/compliance

- Patrick
