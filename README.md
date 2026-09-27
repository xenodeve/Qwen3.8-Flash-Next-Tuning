# Qwen3.8-Flash-Next Tuning

Measurement project for **Qwen3.8-Flash-Next** (MoE, 512 experts × 48 layers) on one desktop:
RTX 5060 Ti 16 GB (PCIe 4.0 x4) + RTX 4070 SUPER 12 GB (PCIe 4.0 x16, display card — keep ≥2.5 GB free) +
i5-13500 (no AVX-512) + 48 GB DDR5-7000 (~108 GB/s), Windows.

Split out of [`xenodeve/Qwen-3.8-27B-Tuning`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning)
on 2026-09-27, because Flash-Next is a different model with different problems (expert
placement, CPU experts, RAM) from the dense 27B.

## Where things are

| | |
|---|---|
| Plan / PRD | issue #1 (Strata-xeno), three-tier design #2, Phase 0 baseline #3 (Strata chosen 2026-09-27) |
| Phase issues | #10–#18 = Phases 1–9, each paired with engine issues Strata-xeno #1–#9 |
| Engine work | [`xenodeve/Strata-xeno`](https://github.com/xenodeve/Strata-xeno) (private fork of Niko1221/Strata) — each engine issue has a twin here that holds the measurements and the decision |
| Retired | [`xenodeve/exllamav3-xeno`](https://github.com/xenodeve/exllamav3-xeno) — reference only; its twins #4–#9 are closed as superseded |
| Research | `docs/research/strata-mechanisms-2026-09-27.md` on branch `research/strata-mechanisms-source-audit-20260927` — why Strata decodes faster than EXL3 |
| llama.cpp history | `docs/results/31–33`, `docs/reports/40–41`, `docs/plans/`, `qwen38-tuning/results/flash-next-optimization/` — copied from the 27B repo, see `PROVENANCE.md` |

## Shared knowledge (lives in the 27B repo, not copied)

These apply here unchanged; read them there so there is one copy:

- [`docs/reports/CORRECTIONS.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/reports/CORRECTIONS.md) — claims later contradicted by our own data
- [`docs/agents/traps.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/agents/traps.md) — ways of working that produced plausible wrong numbers
- [`docs/reports/04-MEASUREMENT-METHODOLOGY.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/reports/04-MEASUREMENT-METHODOLOGY.md) — pairing, ABBA, noise floors
- The EXL3 Claude Code server layer (`qwen38-tuning/serving/exl3/`) stays there for the 27B; Strata-xeno ports its behaviour (Phase 6, #15).

The launchers and bench tests copied here still run from `C:\AI` (the hub). The copies are
the record. Changes to live launchers happen in the 27B repo until the hub is split.
