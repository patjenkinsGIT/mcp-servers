# Let's Encrypt's CP/CPS Was Missing a Required Attestation — And No, Your Certificates Are Fine

On August 10, 2026, Let's Encrypt disclosed that its Certificate Policy/Certification Practice Statement (CP/CPS) did not contain an explicit attestation of compliance with the latest Chrome Root Program Policy and the CCADB Policy — an attestation required by June 15, 2026 under Section 1.1.3 of Chrome Root Program Policy v1.8.

Because Let's Encrypt sits behind a very large share of the web's TLS certificates, the question that immediately followed was whether this forces mandated revocation. It does not. That answer is now on the record from Let's Encrypt itself, and the rest of this post explains why — because the reasoning is more useful than the incident.

## What happened

Root programs require CAs to state, in their CP/CPS, that they comply with the current versions of the governing policies. Chrome Root Program Policy v1.8 set June 15, 2026 as the effective date for that explicit attestation. Let's Encrypt's CP/CPS did not carry it. Let's Encrypt had been tracking the work internally and missed the date, then disclosed the gap publicly on its community forum on August 10, 2026.

The incident is tracked in Mozilla Bugzilla as bug **2062418**, "Let's Encrypt: CPS missing root program attestation."

> A note on a wrong number in circulation: some early write-ups of this story cited Bugzilla 2038351. That is a different, unrelated Let's Encrypt incident — Gen Y cross-certified subordinate CAs missing the serverAuth EKU, from May 2026. If you see 2038351 attached to this story, it's the wrong bug.

## Does this mean certificate revocation?

No. Asked that question directly on the follow-up thread on August 18, 2026, Let's Encrypt answered:

> "Certificate revocation is only required when certificates were issued in violation of the CPS or other relevant requirements. The CPS itself being in violation of requirements does not affect the trust status of any certificates."

That is the CA stating its position on the record, and it settles the only question that had operational consequences for certificate holders.

## Why a document defect is not a certificate defect

This is the part worth keeping, because some version of it recurs every time a CA files an incident report.

A CP/CPS is the document in which a CA describes what it does and commits to doing it. A certificate is a thing the CA issued. The two fail in different ways:

- **The document is defective.** The CA's disclosure doesn't say something it was required to say. The remedy is to fix the document and file an incident report. The certificates that were issued during the gap were still issued according to practice — nothing about them changed when the omission was noticed, and nothing changes about them when the text is added.
- **The certificates are defective.** Something was issued in violation of the requirements — a bad field, a disallowed EKU, an over-long validity period, a name that shouldn't have been signed. That is mis-issuance, and revocation timelines attach to it.

This incident is the first kind. Reading it as the second is what produces an unnecessary fire drill.

The reliable question to ask, before touching anything, is: **does the defect touch the certificates, or only the paperwork about them?** If a CA's own incident report describes a document that needs updating, the answer is usually the second — and your inventory is not involved.

## Who is affected

- **If you hold Let's Encrypt certificates:** no action. Your certificates' trust status is unchanged. There is nothing to inventory, nothing to rotate, and no date on your calendar from this.
- **If your PKI is entirely from other CAs:** no action, and no need to check whether stray Let's Encrypt certificates exist in dev or staging *for this reason*. (Knowing your full certificate inventory is good practice on its own merits — it just isn't something this incident creates a reason to do.)
- **If you operate a CA:** this one is worth reading properly. The attestation requirement in Section 1.1.3 applies to you too, and the June 15, 2026 effective date has passed. Confirm your own CP/CPS carries it.

## The dates

- **2026-06-15** — Effective date of the CP/CPS attestation requirement under Section 1.1.3 of Chrome Root Program Policy v1.8. This date is in the past, and it bound CAs, not certificate holders.
- **2026-08-10** — Let's Encrypt discloses the gap on its community forum.
- **2026-08-18** — Let's Encrypt confirms on the record that no certificate revocation follows.

There is no fourth date. This incident creates no deadline for anyone who simply uses certificates.

## What to do now

Nothing to your certificate inventory.

If you want to take something from this, take the distinction: a CP/CPS defect is a CA compliance matter, and a mis-issuance is a certificate matter. Only the second one ever lands in your renewal calendar. Applying that filter to the next CA incident that trends will save you a week of unnecessary work.

We track every PKI compliance deadline that *does* require action from you at [fixmycert.com/compliance](https://fixmycert.com/compliance) — this one didn't make the list, and that's the point.

## Sources

- Let's Encrypt disclosure, 2026-08-10: https://community.letsencrypt.org/t/2026-08-10-cps-missing-root-program-attestation/250212
- Let's Encrypt follow-up on revocation, 2026-08-18: https://community.letsencrypt.org/t/re-2026-08-10-cps-missing-root-program-attestation/250462
- Mozilla Bugzilla 2062418 — Let's Encrypt: CPS missing root program attestation
