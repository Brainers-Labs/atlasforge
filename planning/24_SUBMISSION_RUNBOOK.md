# NAIC Submission Runbook

**Written Thu 8 Oct 2026.** Hard deadline **Mon 12 Oct 2026, 11:59 PM WAT**. The NAIC page is
explicit: *"Late submissions will not be accepted."*

Target is to submit **Sun 11 Oct** and treat Monday as pure buffer. Every step below names an owner
and a deadline. `You` means it needs a person with a browser, a GPU or a signature — no amount of
code closes it. `Claude` means it can be done in the repository.

## The critical path

Three things block everything else, and two of them get harder every day they are not started:

```
HF token + Awarri licence  ──►  live_smoke.py on a GPU  ──►  evidence/live_<date>.md
                                        │                     (component 2)
                                        └──────────────────►  demo video beats 0:55–3:20
                                                              (component 5)

recruit testers  ──►  sessions  ──►  validation/testers/T0X.md
                                     (component 3 — judged criterion 3)
```

Nothing else is on the critical path. Components 4 and the documentation half of 1 are done; the
team profile, endorsement and Colab run are paperwork and an afternoon.

## What must not happen

The NAIC page requires *"Evidence of testing with actual users, real data or live benchmarks. **Not
simulated or hypothetical results.**"* The committed example reports are synthetic and are labelled
as such — they are documentation, not validation. If the GPU run or the tester rounds do not happen,
the submission says so and marks the row `not run`. **A missing row is survivable. A fabricated one
is disqualifying, and it is also the one thing this project has refused to do at every step so far.**

---

## Thu 8 Oct — remove the blockers today

| # | Step | Owner | Deadline | Done when |
|---|---|---|---|---|
| 1 | **Accept the Awarri licence on all five `NCAIR1` repositories** and create an `HF_TOKEN` with read access | You | **5 PM today** | `atlasforge doctor` shows gated access ✓ for all five |
| 2 | **Start the GPU run**: `python scripts/live_smoke.py --out evidence [--quantize 4bit] [--audio-dir clips]` | You | **start tonight** | A file exists in `evidence/`, even if rows say `not run` |
| 3 | **Email NAIC** — the team-size conflict (the track card says 2–5, the FAQ says 1–6 for Innovation & Enterprise) is still unresolved on their own page | You | **today** | Sent; reply saved for `evidence/` |
| 4 | **Invite 5 tester candidates** so that 2 finish. Problem 02/03 teams are the best fit — they are your users | You | **tonight** | 5 messages sent, dates being negotiated |
| 5 | **Run the notebook in Colab**, top to bottom | You | **tonight** | A Colab link that opens and runs, or a screenshot in `evidence/` |
| 6 | Decide the **benchmark-pack licence** question (FLORES-200 is CC-BY-SA-4.0, share-alike, inside an Apache-2.0 repo) | You | **Fri** | Either packs are committed with PROVENANCE, or the decision *not* to ship one is recorded |
| 7 | Sweep the stale `pip install atlasforge` out of `planning/` — it is the wrong package, and it is on screen in the demo script | Claude | today | No bare `pip install atlasforge` left outside historical narrative |

Step 1 is thirty minutes and unblocks steps 2 and 13. Do it before anything else on this list.

## Fri 9 Oct — get the two long poles moving

| # | Step | Owner | Deadline | Done when |
|---|---|---|---|---|
| 8 | **Complete the live run.** Close every `not run` row by hand, or record why it cannot be closed | You | **EOD Fri** | `evidence/live_<date>.md` with repo IDs, commit SHAs, hardware, per-language samples, timings |
| 9 | **Tester Round 1** — a real session with each tester, notes taken live | You | **EOD Fri** | `validation/testers/T01.md`, `T02.md` — dated, with what they ran and where they got stuck |
| 10 | **Team profile**: members, roles, experience, stream ownership | You | **EOD Fri** | A section ready to paste, consistent with the team size NAIC confirms |
| 11 | **Endorsement**: CAC certificate, or the institutional letter | You | **Fri** | A scan you can attach |
| 12 | Point the docs and README at the live evidence, so a judge can find component 2 from the front page | Claude | Fri | Link checked on the live site |

If step 8 cannot happen because there is no GPU, that is a **today** problem, not a Saturday one —
say so now so the submission can be written honestly around the gap.

## Sat 10 Oct — turn rounds into evidence

| # | Step | Owner | Deadline | Done when |
|---|---|---|---|---|
| 13 | **Record the demo video**, per `planning/19_DEMO_STORY.md`, 3–5 minutes | You | **EOD Sat** | Recorded, tokens not visible in any frame |
| 14 | **Tester Round 2** — re-run the same task after fixing what Round 1 found | You | **EOD Sat** | Round 1 issues listed with the fix for each; testers confirm the fix |
| 15 | Write up the measured numbers — time to first eval, success rate, usability. **Only what was actually observed** | You + Claude | Sat | `validation/` holds numbers traceable to a session |
| 16 | Consent recorded for every name, quote or recording used | You | Sat | A line per tester, or nothing of theirs is quoted |
| 17 | Hausa quickstart, **only if** a named native speaker has agreed to review it | You | Sat | Reviewed, or deliberately absent — never unreviewed |

## Sun 11 Oct — freeze and submit

| # | Step | Owner | Deadline | Done when |
|---|---|---|---|---|
| 18 | **Final gate**: `gitleaks`, `pip-audit`, and grep the whole history for `hf_` | Claude | **Sun AM** | All clean; the grep returns only the placeholder in docs |
| 19 | **Every link in the submission opens while logged out** — repo, docs site, PyPI, video, Colab | Claude + You | **Sun AM** | Each one checked in an incognito window, not just assumed |
| 20 | Beta endpoint shut down or restricted | You | Sun | Anything with a public URL is closed or gated |
| 21 | Confirm the release tag and the version the docs claim agree | Claude | Sun | Tag, `__version__` and changelog heading match |
| 22 | Assemble all seven components and **submit** | You | **Sun EOD** | Submitted |
| 23 | Save the confirmation email or screenshot | You | **Sun** | `evidence/submission_confirmation.*` |

## Mon 12 Oct — buffer only

11:59 PM WAT is a wall, not a target. Nothing should be scheduled here. If Sunday slipped, this is
where it gets caught — and steps 18–20 are the only ones that can be done in an hour.

---

## Open questions that are not ours to close

From `planning/02_NAIC_REQUIREMENTS.md`, still unanswered:

- **Team size** — 2–5 or 1–6? The conflict is on NAIC's own page. Email sent (step 3); the team
  profile cannot be finalised until they answer.
- **Accepted format for beta-tester evidence** — asked, not confirmed.
- **Does integration through the official Hugging Face weights satisfy the 15–17 Oct verification?**
  Everything here assumes yes. It is worth having in writing before the submission leans on it.
- **Is a public repository required**, or is private access for judges acceptable?

## Two decisions that change what gets built

1. **`v0.1.0` or stay on `v0.1.0a2`?** The checklist asks for a tag; you have a pre-alpha. A stable
   tag reads better to a judge and would also correct the `--pre` wording frozen onto the PyPI page,
   which no other action can fix. It costs one workflow run.
2. **If the GPU run fails entirely**, the submission ships with component 2 marked `not run` and a
   plain statement of what was and was not exercised. That is a weaker submission. It is also the
   only honest one, and it is consistent with the status page the project already publishes.

## Progress

Track it against `planning/16_SUBMISSION_CHECKLIST.md`, which holds the same items as tick boxes.
When a step here completes, tick the matching box there — that file is what gets read at the end.
