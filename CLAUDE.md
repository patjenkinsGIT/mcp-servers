# CLAUDE.md — mcp-servers

## FixMyCert compliance sessions: the handoff log

At the **start** of every FixMyCert compliance session, read
`/Users/Patrick/Claude/Projects/FixMyCert/HANDOFF-LOG.md`.

At the **end**, append one line per event you caused or observed on the droplet.

Format — UTC, timestamp of the **event** not of writing, newest at the bottom:

```
YYYY-MM-DDTHH:MM:SSZ | CC | TAG | text
```

Tags:

- `DEPLOY` — commit applied, service restarted, deploy verified
- `APPLY` — data written to the repo
- `FINDING` — established from a primary source, with the URL
- `OPEN` — unresolved item, to be closed later
- `CLOSED` — closes a prior `OPEN`
- `CORRECTION` — retracts or amends an earlier line
- `DECISION` — a call Pat made that should stop being re-litigated

### Three rules

1. **Append-only.** Never edit or delete a prior line. Correct by appending a
   `CORRECTION` that names what was wrong.
2. **Every `OPEN` you write gets a later `CLOSED`** restating the subject, so it
   is greppable.
3. **Entries are attributed claims, not verified state.** Treat `COWORK` lines as
   "what Cowork reported" — cite them as such, never restate one as your own
   finding. The log is not a way around the capability boundary; it is a way to
   stop guessing in its absence.

### Why `DEPLOY` and `APPLY` are the point

Cowork has no droplet shell, and `mcp-servers` is not mounted in its sessions.
Deploy and apply facts are therefore **unobtainable to it** by any route except
this file. A missing deploy line is what produced the 14:03 error on 2026-07-31:
a report that the droplet had not pulled `a598e0b` and needed a fast-forward and
restart, when it had pulled and restarted eight seconds after the commit.

`COMPLIANCE-STATE.md` excludes deploy state, pins, commits and counts by charter,
on the grounds that Claude Code reads them from the droplet in seconds. **That
exclusion stands** — this file is where they go instead.
