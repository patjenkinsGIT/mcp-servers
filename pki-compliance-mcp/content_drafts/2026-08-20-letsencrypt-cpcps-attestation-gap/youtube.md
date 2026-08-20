# YouTube publish package - Let's Encrypt's CP/CPS Gap: Why Your Certificates Are Fine

## Title

Let's Encrypt's CP/CPS Gap: Why Your Certificates Are Fine

## NotebookLM audio prompt

Create a NotebookLM audio overview styled as two senior PKI engineers talking to each other, calm and slightly amused that a documentation problem got read as a revocation emergency. Conversational back-and-forth, not a lecture, under 15 minutes total. The tone is explanatory and reassuring throughout — at no point should either speaker suggest the audience is at risk or needs to act on their certificate inventory. Must cover: (1) Let's Encrypt disclosed on August 10, 2026 that its CP/CPS was missing an explicit attestation of compliance with the latest Chrome Root Program Policy and the CCADB Policy, required by June 15, 2026 under Section 1.1.3 of Chrome Root Program Policy v1.8; (2) the incident is tracked as Mozilla Bugzilla bug 2062418 — and note explicitly that some early coverage cited 2038351, which is a different and unrelated Let's Encrypt incident about Gen Y cross-certified subordinate CAs missing the serverAuth EKU; (3) the revocation question was asked directly on August 18, 2026 and answered on the record by Let's Encrypt, quoting them: "Certificate revocation is only required when certificates were issued in violation of the CPS or other relevant requirements. The CPS itself being in violation of requirements does not affect the trust status of any certificates."; (4) the central teaching point, given the most time — the difference between a defect in the CA's policy document and a defect in the certificates it issued: a document defect is remedied by fixing the document and filing an incident report, while mis-issuance is what carries revocation timelines, and only the second kind ever reaches a subscriber's renewal calendar; (5) the practical filter for the next CA incident that trends — ask whether the defect touches the certificates or only the paperwork about them, before touching anything; (6) explicitly state that listeners who hold Let's Encrypt certificates have nothing to inventory, nothing to rotate, and no deadline from this, and that the only date in the story, June 15, 2026, is in the past and applied to the CA rather than to them; (7) a brief note that anyone who actually operates a CA should confirm their own CP/CPS carries the attestation, since that requirement's date has passed. The listener should walk away understanding why this was never their problem, and equipped to triage the next one faster. Do not use the phrase 'years of experience' or reference how long either speaker has worked in the industry.

## NotebookLM visual prompt

BACKGROUND: Dark navy (#0f172a) - solid, no gradients. ACCENT COLOR: Red #ef4444 (compliance). STYLE: Clean iconography, minimal clutter, dark mode native. TEXT: White #ffffff primary, gray #94a3b8 secondary. DO NOT: busy backgrounds, cartoon characters, 3D effects, stock photos. Also avoid alarm iconography of every kind — no warning triangles, no pulsing or flashing markers, no distress glows; the visuals should read as explanatory, matching a story whose answer is 'no action required'. Additional content-specific elements: (1) a side-by-side contrast panel, which is the core visual of the piece — left side a document icon labelled 'CP/CPS' with a red accent outline and the caption 'defect here = CA fixes the document', right side a certificate icon in plain white with the caption 'defect here = revocation timelines', and a clear divider between them making the point that this incident sat entirely on the left; (2) a clean horizontal timeline strip marking June 15, 2026 (attestation due, CA-side), August 10, 2026 (Let's Encrypt discloses), and August 18, 2026 (Let's Encrypt confirms no revocation) — the last marker resolved and settled in appearance, not open-ended; (3) a simple two-column icon comparing 'you hold LE certs = no action' vs 'you operate a CA = check your own CP/CPS'.

## Description

Let's Encrypt disclosed on August 10, 2026 that its CP/CPS was missing an attestation of compliance with the Chrome Root Program Policy and CCADB Policy, required by June 15, 2026. The obvious next question was whether that forces certificate revocation. It doesn't — Let's Encrypt confirmed as much on the record on August 18. This video explains why, and why the distinction between a defective policy document and a defective certificate is worth knowing before the next CA incident trends.

🔑 Key Points:
- What the Section 1.1.3 attestation requirement is, and what Let's Encrypt missed
- The revocation question, asked and answered on the record: no
- Why a CP/CPS defect is not a certificate defect — the distinction that matters
- If you hold Let's Encrypt certificates: nothing to inventory, nothing to rotate
- The incident is Bugzilla 2062418 — not 2038351, which is a different incident
- If you operate a CA: the attestation requirement applies to you and its date has passed

📚 Full written guide: https://fixmycert.com/compliance
🔗 More PKI education: https://fixmycert.com

#SSL #TLS #Certificates #PKI #CyberSecurity #DevOps #SRE #LetsEncrypt #ChromeRootProgram #CCADB

## Pinned comment

📌 The short version: this one needs nothing from you.

🟢 If you hold Let's Encrypt certificates — no action. Trust status unchanged, nothing to rotate, no deadline.
🟢 If you operate a CA — check that your own CP/CPS carries the Section 1.1.3 attestation.

Timeline (no reader deadline in this story):
• 2026-06-15 — attestation due under Chrome Root Program Policy v1.8 §1.1.3. Past, and CA-side.
• 2026-08-10 — Let's Encrypt discloses the gap.
• 2026-08-18 — Let's Encrypt confirms no certificate revocation follows.

Incident report: Mozilla Bugzilla 2062418. (Early coverage citing 2038351 has the wrong bug — that's the unrelated Gen Y serverAuth EKU incident.)

Deadlines that *do* need action from you: https://fixmycert.com/compliance

## Thumbnail prompt

BACKGROUND: Solid dark navy (#0f172a). LEFT 60%: bold white text 2 lines max, heavy sans-serif — Line 1: 'LET'S ENCRYPT CP/CPS GAP' Line 2: 'NO REVOCATION'. Line 2 should read as the settled answer, not a question — no question mark anywhere on the thumbnail. RIGHT 40%: a single document icon with a red #ef4444 outline, paired with a clean white certificate icon marked with a simple check, making the document-vs-certificate contrast readable at thumbnail size. NO faces, busy backgrounds, gradients, small text, warning triangles, alarm glows, or more than 2 colors + white.
