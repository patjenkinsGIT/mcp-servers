Let's Encrypt disclosed on August 10, 2026 that its CP/CPS was missing an explicit attestation of compliance with the Chrome Root Program Policy and the CCADB Policy — required by June 15, 2026 under Section 1.1.3 of Chrome Root Program Policy v1.8.

The question everyone asked was whether this forces certificate revocation. It doesn't, and that is now on the record. Asked directly on August 18, Let's Encrypt answered:

"Certificate revocation is only required when certificates were issued in violation of the CPS or other relevant requirements. The CPS itself being in violation of requirements does not affect the trust status of any certificates."

That distinction is the whole story, and it's worth internalizing because it comes up every time a CA files an incident:

A CP/CPS is the document where a CA describes what it does. A certificate is the thing it issued. When the document is defective, the CA fixes the document and files an incident report — here, Bugzilla 2062418. When certificates were issued in violation of requirements, that's mis-issuance, and revocation timelines attach to it. This was the first kind, not the second.

So: if you run Let's Encrypt certificates, there is nothing to inventory, nothing to rotate, and no deadline on your calendar from this. The only date in the story is June 15, 2026 — it's in the past, and it bound the CA, not you.

The useful habit isn't scanning your inventory every time a CA incident trends. It's asking one question first: does this defect touch the certificates, or only the paperwork about them?

Live compliance tracker: https://fixmycert.com/compliance

#PKI #TLS #CertificateManagement #LetsEncrypt #Compliance
