# Front-end spec: "In force" badge for `status: "ongoing"` deadlines

**Status: backend live (dataVersion 2.4.9, 2026-07-21) — front-end NOT yet implemented.**
Until the Replit change ships, 14 entries carry correct data but still render a
green "Completed" badge. This spec is the handover for that change.

## Problem

The Compliance Hub badge for past-dated deadlines is derived purely from the
date: past → green "✓ Completed". That is semantically wrong for entries where
the date *started* a rule that is still in force — e.g. the Microsoft April
2026 CTL NotBefore distrust, where renewals under affected roots fail Windows
chain validation **today**. A green check tells the reader "handled, nothing
for me," which is the exact misdiagnosis the entry text warns against.

## Field contract

- Field: `status` on each deadline object in `/api/compliance-data` (and
  `/deadlines`, `/api/compliance/deadlines`, the CSV export).
- Values the front-end will see: `"upcoming"`, `"passed"`, `"ongoing"`.
  (`"ongoing"` is hand-classified in the backend and never expires by date;
  everything else stays date-computed. No other values are sent.)

## Required mapping

| `status` | Badge label | Color | Notes |
|---|---|---|---|
| `"ongoing"` | **In force** | Amber (same family as the existing 31–60-day "High" amber) | NEW — takes precedence over all date-derived badge logic |
| `"passed"` | Completed | Green (unchanged) | current behavior |
| `"upcoming"` | current date-derived rendering (Urgent/High/Upcoming) | unchanged | unchanged |

The ONLY change: when picking the badge for a card or timeline row, check
`status === "ongoing"` FIRST and short-circuit; otherwise keep the existing
date-derived logic exactly as is. Do not change grouping/placement — ongoing
entries keep sorting by date into "Recently Occurred" / "Show Past" (this is
already how the two previously-ongoing entries behave; only their badge is
wrong).

Days-ago text ("62d ago") stays as is — the date is still real; only the
completion claim is wrong.

## Legend reconciliation (do not add a third label)

The timeline legend already advertises an **"In Progress"** state that never
renders anywhere in the DOM (checked 2026-07-21: zero instances). Whoever
wires this up should reuse/rename that legend slot to **"In force"** with the
amber swatch, rather than leaving a dead legend entry and adding a new one.

## Entries currently carrying `status: "ongoing"` (14, as of 2.4.9)

chrome-entrust-distrust (2024-11-11), apple-entrust-distrust (2024-11-15),
mozilla-entrust-distrust (2024-11-30), microsoft-entrust-distrust (2025-04-16),
chrome-clientauth-ica (2025-06-15), dora-effective (2025-01-17),
nis2-germany-bsi (2025-12-06), digicert-g1-root-distrust (2026-04-15),
luxembourg-nis2-in-force (2026-05-10), microsoft-april-2026-ctl-notbefore
(2026-05-19), microsoft-secure-boot-expiry (2026-06-01),
chrome-dedicated-tls-enforcement (2026-06-15),
chrome-digicert-legacy-roots-distrust (2026-07-01),
microsoft-trp-single-purpose-roots (2026-07-01).

Two of these (`dora-effective`, `nis2-germany-bsi`) come from framework
sub-lists — the unified API flattens them identically, but if any framework
detail view renders its own status badges, apply the same mapping there.

Don't hardcode this list client-side — render from the field. The
classification rule (documented above `DEADLINES` in `pki_compliance_mcp.py`):
type (b) = the date started a rule whose failure mode recurs per-reader after
that date (renewal traps, rolling processes, in-force regimes) → ongoing;
type (a) = the ecosystem transitioned once and the rule is now baseline →
computes passed.

## Acceptance check

On /compliance with "Show Past" enabled: the Microsoft April 2026 NotBefore
card shows an amber "In force" badge (not "✓ Completed"); the SC099
validation-logging card (type a, 2026-07-15) still shows green "Completed";
the legend shows "In force" and no orphaned "In Progress".
