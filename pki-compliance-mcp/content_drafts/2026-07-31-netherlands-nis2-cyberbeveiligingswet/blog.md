# Netherlands NIS2 Law (Cyberbeveiligingswet) Takes Effect August 15, 2026

On July 7, 2026, the Dutch Senate approved the Cyberbeveiligingswet — the Netherlands' transposition of the EU's NIS2 Directive — alongside the Wet weerbaarheid kritieke entiteiten (the CER Directive transposition). The Cyberbeveiligingswet enters into force on August 15, 2026. This is not a certificate-specific regulation, but for enterprise certificate teams with Dutch entities or operations, it changes what counts as compliance evidence and what your incident-reporting process needs to cover.

## What happened

The Dutch Senate passed two laws on July 7, 2026:

- **Cyberbeveiligingswet** — transposes NIS2, establishing a three-tier regulatory regime for in-scope entities.
- **Wet weerbaarheid kritieke entiteiten** — transposes the CER Directive, covering critical entity resilience.

The Cyberbeveiligingswet enters into force on **August 15, 2026**. As of the Senate vote, formal notification of the transposition had not yet been submitted to the European Commission. Separately, the Commission referred the Netherlands — along with Ireland, Spain, and France — to the CJEU on July 8, 2026 over NIS2 non-transposition. That referral landed one day after the Senate vote; how it interacts with the now-passed law is a legal detail worth watching, not something we're going to speculate on here.

## Who is affected

This law applies to Dutch entities classified as "essential" or "important" under NIS2's three-tier regime — it is a broad cybersecurity and incident-reporting law, not a PKI-specific one. It affects your organization if you operate legal entities, subsidiaries, or in-scope services within the Netherlands and fall into one of NIS2's covered sectors (energy, health, digital infrastructure, manufacturing, and others per the Directive's scope).

**If you have no Dutch entities or operations, this specific law does not apply to you.** No action is needed here unless a future post covers your jurisdiction's own NIS2 transposition — check our tracker for those as they land.

For teams that are in scope: NIS2's risk-management and incident-reporting obligations increasingly touch certificate lifecycle management. Mis-issuance, key compromise, revocation failures, and outage-causing expirations are exactly the kind of "significant incident" categories regulators expect you to detect and report on. Your cert inventory and key-management logs are becoming compliance artifacts, not just operational hygiene.

## Key dates

- **July 7, 2026** — Dutch Senate approves the Cyberbeveiligingswet (NIS2) and Wet weerbaarheid kritieke entiteiten (CER).
- **July 8, 2026** — European Commission refers the Netherlands (with Ireland, Spain, France) to the CJEU over NIS2 non-transposition.
- **August 15, 2026** — Cyberbeveiligingswet enters into force in the Netherlands.

## What to do now

1. **Confirm scope.** Determine whether any of your Dutch entities, subsidiaries, or in-country operations fall under NIS2's essential or important entity classification under the new three-tier regime.
2. **Check your incident-reporting taxonomy.** If in scope, make sure certificate-related incidents — key compromise, mis-issuance, failed revocation, expiry outages — are explicitly captured in whatever incident classification feeds your NIS2 reporting obligations starting August 15.
3. **Assemble the evidence trail now.** Cert inventory records, key-management policies, and revocation/incident logs are the kind of technical and organizational measures documentation regulators will expect during oversight. Don't wait for a request to find out your evidence has gaps.
4. **Watch the notification status.** Formal notification of the transposition to the European Commission had not been submitted as of the Senate vote — track confirmation that it's been lodged, since that affects enforcement clarity going forward.
5. **If you have no Dutch footprint, stand down on this specific item** — but keep monitoring, since other EU member states are at various stages of their own NIS2 transposition and referral status.

This is a fast-moving regulatory season across the EU. Track this and every other PKI-relevant compliance deadline on the live tracker at [fixmycert.com/compliance](https://fixmycert.com/compliance).
