# CLAUDE.md

Same operating rules as `xenodeve/Qwen-3.8-27B-Tuning` (read its CLAUDE.md, CORRECTIONS.md and
traps.md before quoting any number). Chat Thai; reports, code, commits English; tracker bodies
bilingual. Engine changes go to `xenodeve/Strata-xeno` (public) with a paired issue here; `exllamav3-xeno` is
reference only. The 4070 SUPER is the display card: other processes get 2.5 GB (2560 MiB) of it in total, **including** the
desktop's current use — not 2.5 GB free on top of it.

**Measure latency, never guess where the time went.** Any claim about why decode or prefill moved (or did
not) must cite the per-stage counters of the same paired runs — `verify window`, `pool multi`, `dispatch`,
`dispatch detail`, `secondary timing`, `mtp`, `tier hits`, parsed with Strata's `tests/xeno/latency_breakdown.py`
— compared stage by stage A vs B. If a stage is not instrumented, add a timer. Estimates stitched from
different runs or a profiler trace are hypotheses and must say so. Full rule: `AGENTS.md` in Strata-xeno.

**Measure at high CPU priority, every time.** Other programs take CPU from Strata's pinned pool workers and cause
random slow runs. Every A/B, ABBA or sweep passes `--pool-priority 2` in every arm (Strata-xeno `AGENTS.md`), records
it in the report, and runs no builds or other CPU-heavy work while a measurement is in flight.
