#!/usr/bin/env python3
"""Offline tests for the Stripe webhook helpers in pki_compliance_mcp.py.

Covers signature verification (valid, tampered, expired, malformed,
multiple v1 candidates), event summarization, and Pushover no-op without
credentials. No network calls.
"""
import hashlib
import hmac
import json
import os

import pki_compliance_mcp as pki

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


SECRET = "whsec_testsecret"
NOW = 1_784_200_000


def sign(payload: bytes, ts: int = NOW, secret: str = SECRET) -> str:
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + payload,
                   hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


print("== verify_stripe_signature ==")
body = b'{"id": "evt_1"}'
check("valid signature accepted",
      pki.verify_stripe_signature(body, sign(body), SECRET, now=NOW))
check("tampered payload rejected",
      not pki.verify_stripe_signature(b'{"id": "evt_2"}', sign(body), SECRET, now=NOW))
check("wrong secret rejected",
      not pki.verify_stripe_signature(body, sign(body, secret="whsec_other"),
                                      SECRET, now=NOW))
check("expired timestamp rejected",
      not pki.verify_stripe_signature(body, sign(body, ts=NOW - 600),
                                      SECRET, now=NOW))
check("future timestamp rejected",
      not pki.verify_stripe_signature(body, sign(body, ts=NOW + 600),
                                      SECRET, now=NOW))
check("empty header rejected",
      not pki.verify_stripe_signature(body, "", SECRET, now=NOW))
check("malformed header rejected",
      not pki.verify_stripe_signature(body, "garbage", SECRET, now=NOW))
good = sign(body)
check("extra bogus v1 still accepted",
      pki.verify_stripe_signature(body, good + ",v1=deadbeef", SECRET, now=NOW))
check("only bogus v1 rejected",
      not pki.verify_stripe_signature(body, f"t={NOW},v1=deadbeef", SECRET, now=NOW))

print("== summarize_stripe_event ==")
SALE = {
    "id": "evt_sale",
    "type": "checkout.session.completed",
    "data": {"object": {"amount_total": 19900, "currency": "usd",
                        "customer_details": {"email": "buyer@example.com"}}},
}
s = pki.summarize_stripe_event(SALE)
check("sale event summarized", s is not None)
check("amount formatted", "199.00 USD" in s["message"])
check("buyer email included", "buyer@example.com" in s["message"])
check("title is the sale banner", "sale" in s["title"].lower())

check("other event types ignored",
      pki.summarize_stripe_event({"type": "invoice.paid", "data": {}}) is None)
s2 = pki.summarize_stripe_event({
    "type": "checkout.session.completed", "data": {"object": {}}})
check("missing amount degrades gracefully", "unknown amount" in s2["message"])
check("missing email degrades gracefully", "unknown buyer" in s2["message"])

print("== send_pushover ==")
os.environ.pop("PUSHOVER_TOKEN", None)
os.environ.pop("PUSHOVER_USER", None)
check("no-op without credentials",
      pki.send_pushover("t", "m") is False)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
