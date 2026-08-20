<!-- OPERATOR NOTES - do not send as-is:
     audience: FixMyCert brand ONLY (brand:fmc, tag 19302998) - NOT the full list
     sender:   Patrick from FixMyCert <patrick@fixmycert.com> (account default)
     This is a DRAFT. Create in Kit as a draft broadcast; never auto-send. -->

## Subject

Let's Encrypt's CP/CPS gap: no, your certs are fine

## Preview text

The revocation question got answered on the record — and the answer is no action

## Body

Hey — short one, and it's good news wearing a scary headline.

**What happened:** Let's Encrypt disclosed on August 10 that its CP/CPS was missing an explicit attestation of compliance with the Chrome Root Program Policy and the CCADB Policy, required by June 15 under Section 1.1.3 of Chrome Root Program Policy v1.8. The incident is tracked in Bugzilla 2062418.

**Does it mean revocation?** No. Asked point-blank on August 18, Let's Encrypt answered on the record: "Certificate revocation is only required when certificates were issued in violation of the CPS or other relevant requirements. The CPS itself being in violation of requirements does not affect the trust status of any certificates."

**Why that's the whole story:** a CP/CPS is the document where a CA describes what it does. A certificate is the thing it issued. When the document is defective, the CA fixes the document and files an incident report. When certificates were issued in violation of requirements, that's mis-issuance — and that's when revocation timelines show up. This was the first kind.

**What you should do:** nothing. Your Let's Encrypt certificates' trust status is unchanged. Nothing to inventory, nothing to rotate, no date to put on the calendar. The only date in this story is June 15, 2026 — it's past, and it applied to the CA, not to you.

The one thing worth keeping is the filter. Next time a CA incident starts trending, ask first: does this defect touch the certificates, or only the paperwork about them? Only the first kind ever reaches your renewal calendar.

If you do run a CA yourself, that attestation requirement applies to you too and its date has passed — worth a look at your own CP/CPS.

Everything that *does* need action from you is on the tracker:

https://fixmycert.com/compliance

- Patrick
