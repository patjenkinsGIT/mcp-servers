Let's Encrypt disclosed on August 10, 2026 that its CP/CPS is missing a required attestation of compliance with the Chrome Root Program Policy and CCADB Policy — a deadline that passed on June 15, 2026.

The community (and a Mozilla Bugzilla thread) is now debating whether this gap requires mandated certificate revocation. As of today, that has not been confirmed either way.

Why this matters even before it's resolved: Let's Encrypt issues an enormous share of the web's TLS certificates via ACME automation. If you have any certs issued by ISRG Root X1/X2 or intermediates R3/R10/R11, you should know that right now — not after a decision drops.

If your PKI is entirely from other CAs, you likely need no action here. If you're not sure, that's the point of this post.

What we're doing: inventorying LE-issued certs, checking ACME automation dependencies, and holding off on any rotation until there's an actual ruling.

Are you finding Let's Encrypt certs in places you didn't expect?

Live tracker: https://fixmycert.com/compliance

#PKI #TLS #CertificateManagement #LetsEncrypt #Compliance
