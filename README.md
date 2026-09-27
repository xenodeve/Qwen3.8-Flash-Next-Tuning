# Qwen3.8-Flash-Next Tuning

Measurement project for **Qwen3.8-Flash-Next** (MoE, 512 experts × 48 layers) on one desktop:
RTX 5060 Ti 16 GB (PCIe 4.0 x4) + RTX 4070 SUPER 12 GB (PCIe 4.0 x16, display card) +
i5-13500 (no AVX-512) + 48 GB DDR5-7000 (~108 GB/s), Windows.

Split out of [`xenodeve/Qwen-3.8-27B-Tuning`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning)
on 2026-09-27, because Flash-Next is a different model with different problems (expert
placement, CPU experts, RAM) from the dense 27B.

## Where things are

| | |
|---|---|
| Plan / PRD | issue #1 (EXL3-xeno), three-tier design #2, substrate bake-off #3 |
| Engine work | [`xenodeve/exllamav3-xeno`](https://github.com/xenodeve/exllamav3-xeno) — each engine issue has a twin here that holds the measurements and the decision |
| llama.cpp history | `docs/results/31–33`, `docs/reports/40–41`, `docs/plans/`, `qwen38-tuning/results/flash-next-optimization/` — copied from the 27B repo, see `PROVENANCE.md` |

## Shared knowledge (lives in the 27B repo, not copied)

These apply here unchanged; read them there so there is one copy:

- [`docs/reports/CORRECTIONS.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/reports/CORRECTIONS.md) — claims later contradicted by our own data
- [`docs/agents/traps.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/agents/traps.md) — ways of working that produced plausible wrong numbers
- [`docs/reports/04-MEASUREMENT-METHODOLOGY.md`](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/blob/main/docs/reports/04-MEASUREMENT-METHODOLOGY.md) — pairing, ABBA, noise floors
- The EXL3 Claude Code server layer (`qwen38-tuning/serving/exl3/`) stays there and serves both models.

The launchers and bench tests copied here still run from `C:\AI` (the hub). The copies are
the record. Changes to live launchers happen in the 27B repo until the hub is split.
