# Strata mechanisms worth transferring to EXL3-xeno

**Date:** 2026-09-27  
**Scope:** source audit, not a benchmark.  
**Intended repository path:** `docs/research/strata-mechanisms-2026-09-27.md`  
**Tracking:** [Flash-Next #1](https://github.com/xenodeve/Qwen3.8-Flash-Next-Tuning/issues/1)

## Executive finding

Strata's relevant advantage is not adequately described by either “C++ instead of Python” or “AVX-512 instead of AVX2.” The inspected source combines a graph spanning the layer chain, a mapped-memory CPU hand-off, GPU computation of a configurable share of RAM-resident misses during speculative verification, and a simpler Q2_0 weight decoder. These are separable mechanisms, not a demonstrated sixfold speedup for our fork.

The source audit changes four earlier interpretations:

1. The legacy per-layer `session_loop` is not the only execution path. There is an eligible **whole-token graph**, and serving uses a **whole-window verifier graph**.
2. The `ExpertCache` admission class's “no eviction” description is not a description of the complete runtime. The serving driver performs **adaptive replacement between verification rounds**.
3. EXL3 already has **asynchronous issue/collect** and multi-token expert grouping. We should identify overhead around those mechanisms, not propose them as missing from scratch.
4. The local AVX-VNNI experiment is already implemented. Its reported gate/up improvement is about **13%**, not the earlier hypothetical 25–80%; state extraction and `mul1` reconstruction remain separate work.

Evidence and limitations follow. None of the proposed speedups has been measured by this research session.

## 1. Provenance and access boundary

| Label | Source actually inspected | Meaning |
|---|---|---|
| S0 | `Niko1221/Strata@6da1f667e86558b152ab128edf3ebf77a80a9e57` | Primary source snapshot named in the handoff. Its commit describes engine v0.1.2. |
| S1 | `Niko1221/Strata@11bb7e293ff295839661275dcee5e99babf06410` | Used only for the explicitly identified PLE-reader header. |
| X0 | `xenodeve/exllamav3-xeno@12414d0af7b3beeabdda5990f6b554b996fa1416` | Accessible published baseline; not the unpublished local Phase-B branch. |
| H0 | User-supplied `handoff-strata-research(1).md`, lines 35–45 | Local observations supplied by the building session; not rerun here. |
| P0 | Strata PR metadata read on 2026-09-27 | Status of PRs #7–#10, independently checked. |

The handoff identifies a local prebuilt engine 0.1.3 beside an S0 source tree. This audit does **not** certify that the binary was compiled from S0. Likewise, the public fork exposed `master` at X0 and a `dev` branch, but not `xeno/phase-b-remote-slice`; the research environment cannot read `D:\Github\exllamav3-xeno`.

Consequently, the requested exact `file:line` comparison against the **local modified fork** is not fully attainable here. Every EXL3 line reference below is explicitly X0. Local changes to `xeno_*`, `expert_tiers.py`, and `Isa::AvxVnni` are described only where H0 or the engine tracker reports them. No local line numbers are invented.

Repository reads succeeded through the GitHub connector. A shell clone attempt failed because this environment could not resolve `github.com`; that does not invalidate the pinned file reads, but it means a local clone was not obtained. No models or GPU workloads were run, no processes were killed, and no engine implementation was changed.

### Carried-forward observations, not experiments to repeat blindly

H0 reports approximately 10–14 generated tokens/s; roughly 4 ms of CPU work per layer-token in an **82%-of-experts-on-CPU configuration**, with GEMV accounting for 83%; and about 3 GB/s effective expert processing. H0 separately reports that frequency-ordered placement reduces CPU selections to about 12%, that 4–6 P-core threads beat 10 workers, that a remote `forward()` formerly cost about 480 microseconds per layer, and that a direct fused call reduced this cost.

These observations refer to different experiment configurations and must not be summed into a single current timing budget. H0 also records background CPU contention. The [engine #4 update](https://github.com/xenodeve/exllamav3-xeno/issues/4#issuecomment-5853093950) reports AVX-VNNI gate/up time of approximately 1930 → 1674 microseconds in its paired test and explicitly excludes contaminated rounds. This is useful evidence, not a license to apply llama.cpp's previous 1.8× result to EXL3.

### Issue map used below

| Measurement tracker | Engine tracker | Existing work |
|---|---|---|
| Flash-Next #4 | EXL3-xeno #1 | Build/baseline |
| Flash-Next #5 | EXL3-xeno #2 | Static three-tier / remote slice |
| Flash-Next #6 | EXL3-xeno #3 | Asynchronous cross-device hand-off |
| Flash-Next #7 | EXL3-xeno #4 | AVX-VNNI CPU kernel |
| Flash-Next #8 | EXL3-xeno #5 | CPU thread policy |
| Flash-Next #9 | EXL3-xeno #6 | Exclusive rotating tier |

The existing persistent 5060 Ti compute tier, 4070 SUPER streaming tier, ownership policy, and correctness gates are inputs to this report, not newly discovered recommendations.

## 2. Findings

### F01 — Whole-token capture exists beyond the legacy per-layer loop

**Mechanism.** S0 records all model layers into one `TokenGraph` for an eligible non-speculative path, then replays that graph while a native host loop services CPU requests. This removes repeated layer-graph submission from that path but does not remove the CPU dependency inside each MoE layer.

**Evidence.** [S0 `src/core/session.cpp:794–965`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/session.cpp#L794-L965) contains `session_capture_token`, the layer loop inside capture, and `session_run_token` with one graph launch. [S0 `src/program/generate.cpp:1770–1800`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L1770-L1800) selects it subject to capture, dump, hit-table and pack eligibility and prints **“48 layers, one launch per token”**. It is not unconditional for every pack or debug mode.

The older `session_loop` at [S0 `src/core/session.cpp:536–720`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/session.cpp#L536-L720) still launches pre/post graphs per layer. Reading only that function or the older “one graph per layer” header commentary produces the wrong conclusion about the available fast path.

**EXL3 comparison.** [X0 `exllamav3/model/model_ls.py:340–354`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/model/model_ls.py#L340-L354) iterates `self.fwd_modules` and calls `module.prepare_for_device` and `module.forward` in Python. This identifies the outer dispatch seam, not proof that inner modules lack CUDA graphs. The enclosing execution of the unpublished remote-slice branch remains uninspected.

**Port / issue.** **L**, Flash-Next #5/#6 → engine #2/#3: prototype a capture-ready native execution segment around the existing kernels and hand-off, with frozen ownership and preallocated staging. Extend the graph boundary only when the measured remaining submission cost justifies it.

**Register.** `verified-in-source`: Strata path and X0 outer dispatch. `hypothesis`: that eliminating the remaining local dispatch is the largest end-to-end improvement.

### F02 — Serving uses an entire speculative-window graph

**Mechanism.** Strata's serving driver passes the anchor token plus draft tokens through a verifier that captures the complete layer chain and the output head for each supported window width. It commits only the accepted prefix afterward, separating recurrent-state advancement from speculative evaluation.

**Evidence.** [S0 `include/strata/core/verify.hpp:1–22`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/include/strata/core/verify.hpp#L1-L22) specifies **“ONE captured graph”**. The implementation at [S0 `src/core/verify.cpp:618–740`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/verify.cpp#L618-L740) captures `record_window(T, ...)` and launches `exec_[T]`. [S0 `src/program/generate.cpp:2155–2223`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L2155-L2223) calls `ver.run`, compares drafts with target outputs, calls `ver.commit(a + 1)`, and then drafts again.

The source's `--spec` means total window width, not number of future drafts. The first serving window is T=1. This is a graph of kernels, not evidence of a single persistent inference kernel.

**EXL3 comparison.** [X0 `exllamav3/model/model.py:384–409`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/model/model.py#L384-L409) routes generation/verification through the normal forward path and advances recurrent states afterward; F01 identifies its outer LS loop. The local branch's exact graph coverage at T=1/2/4/8 must be obtained from that session, not inferred from the class names.

**Port / issue.** **L**, Flash-Next #5/#6 and PRD #1 speculation stories. Start with a fixed-width T=1 execution audit, then T=4; reuse EXL3's state/cache semantics rather than porting Strata's tensor layouts.

**Register.** `verified-in-source`: full-window capture and accepted-prefix commit. `hypothesis`: whole-window capture will make our current CPU-offloaded MTP profitable.

### F03 — Synchronization is a mapped-memory handshake, not the absence of waits

**Mechanism.** The Strata graph publishes routing inputs to mapped host buffers and waits on a host-written completion flag while the CPU pool computes its share. The host need not submit the next layer's graph at every boundary, but the next dependent residual still cannot advance until the required expert result exists.

**Evidence.** [S0 `src/core/session.cpp:810–945`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/session.cpp#L810-L945) places `doorbell_wait` and `copy_from_mapped` inside capture, then services the flags from `session_run_token`. [S0 `src/core/verify.cpp:530–576`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/verify.cpp#L530-L576) contains `wait_flag_ge(m_flag_, ring, cs)` before consuming CPU partials. [S0 `src/core/verify.cpp:740–805`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/verify.cpp#L740-L805) services each layer's pool callback, publishes flags, and synchronizes after the window.

The token path explicitly uses a copy kernel instead of a copy-engine node to avoid a WDDM submission behavior noted in its comment. That historical performance note is vendor evidence; it is not yet reproduced on our driver. Neither inspected path proves predictive execution of the next layer's routed experts before that layer's input exists.

**EXL3 comparison.** [X0 `exllamav3/modules/block_sparse_mlp_cpu.py:225–316`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/block_sparse_mlp_cpu.py#L225-L316) already calls `submit_issue_fused` and later `submit_collect_fused`; its comment says **“stream-ordered”**. This contradicts any blanket claim that baseline EXL3 lacks asynchronous CPU hand-off. A blocking local `xeno_*` call, a GPU stream wait, and a host/device-wide synchronization are different hypotheses.

**Port / issue.** **M** for an instrumented existing hand-off; **L** for graph-resident cross-device completion. Flash-Next #6 → engine #3. Measure host API self-time separately from GPU-visible exposed wait; do not count overlapping CPU and GPU time twice.

**Register.** `verified-in-source`: handshakes and existing EXL3 issue/collect. `hypothesis`: which surrounding local operation leaves the dominant bubble.

### F04 — A share of RAM misses is computed on the GPU during verification

**Mechanism.** Strata does not require every nonresident expert in a small verification window to run on the CPU. Its dispatcher divides work into VRAM-hit groups, a configurable PCIe/GPU share of RAM misses, and the remaining CPU groups, publishing GPU work before running the CPU portion.

**Evidence.** [S0 `include/strata/core/expert_source.hpp:65–100`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/include/strata/core/expert_source.hpp#L65-L100) defines `GpuPlanSink` and **“the PCIe share”**. [S0 `src/core/expert_source.cpp:310–433`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/expert_source.cpp#L310-L433) publishes resident and staged groups before calling the pool. [S0 `src/core/verify.cpp:530–560`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/verify.cpp#L530-L560) computes resident groups, obtains the PCIe groups, and waits for the CPU share only before combination.

There are three implemented transfer modes: DMA staging, direct device access to mapped host bytes, and a copy kernel inside the graph. [S0 `src/program/generate.cpp:1860–1875`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L1860-L1875) chooses the auto mode by pack type; comments elsewhere describing a different default must not override this branch.

**EXL3 comparison.** X0's small-batch branch in [the CPU split module:225–249](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/block_sparse_mlp_cpu.py#L225-L249) issues CPU work, while its large-batch branch uses `submit_prefill`. The local three-tier branch already changes ownership and remote computation; what remains unverified is whether it makes this fractional, verify-size decision for otherwise CPU-owned misses.

**Port / issue.** **M–L**, Flash-Next #5/#6/#9 → engine #2/#3/#6. This extends the already-designed 4070 SUPER miss policy; it is not a new justification for using the x16 card. First keep residency fixed and use bounded transient staging, so the experiment measures execution choice rather than simultaneous migration-policy changes.

**Register.** `verified-in-source`: fractional GPU-miss path. `hypothesis`: whether it beats our 4–6-P-core CPU path for T=1 or T=4.

### F05 — Q2_0 unpacking and EXL3 mul1 reconstruction have different costs

**Mechanism.** Strata's Q2_0 CPU path unpacks short integer codes and applies block scales and activation corrections, whereas EXL3 reconstructs trellis states and evaluates the mul1 codebook before its dot products. VNNI accelerates a dot-product component of the latter without removing its state extraction, codebook multiplication, transforms, or scheduling costs.

**Evidence.** [S0 `src/kernels/cpu/q2_avx2.cpp:24–97`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/kernels/cpu/q2_avx2.cpp#L24-L97) implements `unpack64` and `row_multi`: masks, shifts and interleaves produce the codes, followed by AVX2 integer dot products and scaled accumulation. [S0 `src/kernels/cpu/expert.cpp:123–195`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/kernels/cpu/expert.cpp#L123-L195) shows VBMI unpacking and VNNI alternatives; that AVX-512 path is not the i5-13500 path.

**EXL3 comparison.** [X0 `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:35–55`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/exllamav3_ext/cpu/moe_mul1.cpp#L35-L55) gives `bytesum(s * 0x83DCD12D)` in the mul1 identity; [lines 175–197](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/exllamav3_ext/cpu/moe_mul1.cpp#L175-L197) implement scalar state extraction and reconstruction. H0 and engine #4 report that the local VNNI path is already bit-exact and that `vpmulld` is the next suspected limit. This report did not inspect that unpublished assembly or establish its cycle cost.

A fair source comparison is therefore **instruction classes per reconstructed weight**, not simply vector width. A precise instruction count for the compiled i5 kernel requires its disassembly; intrinsic counting is not that measurement. Q2_0 and EXL3 quantizations are different artifacts, so copying a Q2_0 decoder into an EXL3 reader would not preserve the represented weights.

**Port / issue.** **M** for isolated mul1 reconstruction experiments, Flash-Next #7 → engine #4. Reuse the implemented VNNI tier; profile/extract the remaining decoder section and test register blocking or equivalent arithmetic without expanding the complete weight pool. A GSQ/Q2 backend would be a separate artifact/backend decision, outside this direct port.

**Register.** `verified-in-source`: arithmetic difference. `hypothesis`: dominance of `vpmulld` in the local remaining time. The reported 13% improvement is carried-forward local evidence, not a new measurement.

### F06 — Multi-token expert grouping amortizes work, but both engines have it

**Mechanism.** Strata builds one CPU job per distinct expert in a verification window and attaches the token activations routed to that expert. Its multi-row kernels can reuse unpacked weights across those activations, although a wider window can still introduce more distinct misses.

**Evidence.** [S0 `src/core/expert_source.cpp:375–433`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/core/expert_source.cpp#L375-L433) uses `job_of` to collect token entries into `ExpertJobMulti`. [S0 `src/kernels/cpu/q2_avx2.cpp:39–97`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/kernels/cpu/q2_avx2.cpp#L39-L97) unpacks a weight block outside the loop over NT activations. The verifier header's reported union-growth factors are vendor trace observations, not universal constants.

**EXL3 comparison.** [X0 `exllamav3/exllamav3_ext/cpu/moe_mul1.h:66–99`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/exllamav3_ext/cpu/moe_mul1.h#L66-L99) explicitly says **“Tokens are grouped by expert”**. X0's CPP sets `MAX_M = 4` near line 70. The missing question is the actual reuse and exposed cost on the local routed traces, not whether EXL3 knows about grouped experts at all.

**Port / issue.** **S** instrumentation, **M** only if the trace identifies a missing fast path; Flash-Next #6/#7 and PRD #1 speculation. Record distinct CPU experts per window, CPU assignment entries, multiplicity per expert, verifier milliseconds, draft milliseconds and accepted output tokens.

**Register.** `verified-in-source`: grouping in both implementations. `hypothesis`: additional amortization available in local MTP.

### F07 — Adaptive replacement exists above Strata's cache allocator

**Mechanism.** Strata can fill a profile-selected cache and subsequently replace low-use resident experts with more-used missing experts between speculative rounds. The allocator itself need not implement eviction because the driver refills a selected slot and updates the residency tables after transfer completion.

**Evidence.** [S0 `src/program/generate.cpp:1870–1940`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L1870-L1940) defines `apply_pending` and `adapt`, ranks candidates/victims, performs an H2D refill, marks the victim nonresident, and admits the incoming expert through a pending event. The nearby comment **“evicted now”** describes an actual state update, not a proposed feature. [Lines 2175–2210](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L2175-L2210) invoke adaptation alongside commit/draft work.

This is a duplicate-backed cache: a victim remains available from the host source. It is not the user's exclusive RAM↔4070 migration design. H0's “reportedly it does not evict” and my earlier blanket no-eviction claim should be replaced by this narrower allocator-versus-driver distinction.

**EXL3 comparison.** [X0 `exllamav3/modules/block_sparse_mlp_cpu.py:50–95`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/block_sparse_mlp_cpu.py#L50-L95) schedules pending hot/cold sweeps when the generator queue drains. Those lines were read earlier in the source investigation; the unpublished Phase-C policy is not available here. The integration difference to study is Strata's *between-window cadence* versus X0's *between-generation cadence*, not an absence of adaptive placement in EXL3.

**Port / issue.** **M**, Flash-Next #9 → engine #6. Reuse measured router statistics and the existing ownership design; isolate admission/cadence from exclusive-memory migration when comparing policies.

**Register.** `verified-in-source`: driver-level replacement and X0 sweep seam. `hypothesis`: that finer update cadence is useful for this user's sessions.

### F08 — CPU work is subdivided when few experts miss

**Mechanism.** Strata subdivides gate/up and down rows across the worker pool, so one or two missed experts can occupy multiple cores rather than one core per expert. Its flat task counter lets fast workers claim more row tiles, while the host can participate in the drain instead of only waiting.

**Evidence.** [S0 `src/kernels/cpu/pool.cpp:164–305`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/kernels/cpu/pool.cpp#L164-L305) implements atomic claims, row ranges, `run_phase`, and `mtasks_ = 3 * threads`. [S0 `include/strata/kernels/cpu/pool.hpp:104–131`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/include/strata/kernels/cpu/pool.hpp#L104-L131) documents row subdivision and multi-phase counters. The inspected path is not evidence of an explicit Intel P/E-core weighting policy.

**EXL3 comparison.** X0's CPU interface already uses a persistent threaded pool ([`moe_mul1.h:66–99`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/exllamav3_ext/cpu/moe_mul1.h#L66-L99)). H0 has already found 4–6 P-core workers preferable to 10; do not reopen the earlier unsupported recommendation to use all physical cores by default. The remaining comparison is tile granularity and tail time at the *observed low miss counts*.

**Port / issue.** **S–M**, Flash-Next #8 → engine #5, with #7 for arithmetic interactions. Use the recorded routing trace and existing P-core baseline; compare per-layer worker tail, not just aggregate CPU utilization.

**Register.** `verified-in-source`: Strata task subdivision. `hypothesis`: transferable scheduling improvement beyond our existing pool.

### F09 — Prefill grouping and temporary cache-slot borrowing

**Mechanism.** Strata groups prompt tokens by expert, processes them in chunks, and streams nonresident weights through pinned staging for GPU matrix multiplication. It may borrow cache slots for prompt scratch and refill those slots before verification, trading a temporary refill cost for greater decode residency.

**Evidence.** [S0 `include/strata/prefill/prefill.hpp:1–64`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/include/strata/prefill/prefill.hpp#L1-L64) specifies the host ring and exposes `experts_streamed`, `experts_dma`, `experts_resident`, `ms_experts_host`, and `ms_ple`. [S0 `src/program/generate.cpp:1810–1872`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L1810-L1872) sizes the borrowed region and shrinks chunks; [lines 2110–2147](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L2110-L2147) refill it. These are the inspected interface and driver, not a complete audit of every prefill kernel.

**EXL3 comparison.** [X0 `block_sparse_mlp_cpu.py:225–249`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/block_sparse_mlp_cpu.py#L225-L249) already selects streamed prefill, and [X0 `moe_mul1.h:101–114`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/exllamav3_ext/cpu/moe_mul1.h#L101-L114) exposes weight staging. The PRD already assigns RAM streaming to the 4070 SUPER; no new topology recommendation is needed.

**Port / issue.** **S** counters, **M** measured scratch reuse, Flash-Next #5/#9. Keep buffer-lending semantics distinct from the already-planned exclusive tier. This is not the leading explanation for slow decode by itself.

**Register.** `verified-in-source`: interface/driver behavior and EXL3 prefill entry point. `hypothesis`: additional prefill benefit on our machine.

### F10 — PLE contributes a useful telemetry model, not a requested rewrite

**Mechanism.** Strata's PLE reader exposes requested rows, cache hits, deduplicated page reads, bytes, submit delay and blocked collection time independently. This allows an actual I/O stall to be separated from CPU expert work and host submission latency.

**Evidence.** [S1 `include/strata/ngram/ple_reader.hpp:27–83`](https://github.com/Niko1221/Strata/blob/11bb7e293ff295839661275dcee5e99babf06410/include/strata/ngram/ple_reader.hpp#L27-L83) defines `ReaderStats`, `issue`, `collect`, and a bounded latency sample. **“wait_us”** and **“submit_us”** distinguish waiting for data from blocking while submitting reads.

**EXL3 comparison.** [X0 `exllamav3/modules/ngram_embedding.py:12–46`](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/ngram_embedding.py#L12-L46) already describes streamed row gathering; [lines 104–124](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/modules/ngram_embedding.py#L104-L124) include `prefetch_stats`. PLE presence is not itself a missing feature.

**Port / issue.** **S**, PRD #1 story 47; attach to Flash-Next #6 instrumentation or propose a telemetry sub-issue, without opening one in this research task. Copy the measurement vocabulary, not the reader implementation.

**Register.** `verified-in-source`: counters and existing row streaming. `hypothesis`: PLE contributes materially to the current 10–14 tokens/s result; no new timing establishes that.

### F11 — PRs #7–#10 are merged, not still open

**Mechanism.** Strata's subsequent serving fixes close the engine generator while holding the request lock, preserve reusable conversation state, sleep idle CPU workers, and route short prompt additions through verification windows. These improve agent operation and follow-up latency but are not a source-level proof of faster steady-state expert arithmetic.

**Evidence/status, checked 2026-09-27.**

| PR | State returned | Mechanism / evidence anchor | Transfer classification |
|---|---|---|---|
| [#7](https://github.com/Niko1221/Strata/pull/7) | Merged | `serve/server.py`, `Service.run`: finalize/drain the engine generator before releasing `fifo`. | Behavioral test, not a direct EXL3 patch. |
| [#8](https://github.com/Niko1221/Strata/pull/8) | Merged | Conversation checkpoints of GDN/PLE/indexer-tail state; cache-state fingerprints reported by author. | Compare with existing EXL3 cache tests. |
| [#9](https://github.com/Niko1221/Strata/pull/9) | Merged | Spin briefly, then sleep idle workers on a condition variable. | Check behavior already implemented before copying. |
| [#10](https://github.com/Niko1221/Strata/pull/10) | Merged | Short additions through `ver.run → ver.commit → mtp.prefill`; larger chunks stay batched. | Candidate short-read policy test. |

Metadata gives merge time 2026-09-26 20:27:35 UTC for all four. Their implementation patches were not exhaustively audited here; this finding verifies metadata and the authors' described behavior. The handoff's “open PRs” label is stale, and S0 precedes these changes. Reported tests in the PR bodies remain author-reported, not ours.

**EXL3 comparison.** The PRD already carries our EXL3 serving compatibility and session work; X0's `prefill_ls`/`forward_ls` entry points are at [model_ls.py:322–354](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/model/model_ls.py#L322-L354). Exact equivalence to the local server requires its current source. There is no basis for copying a single-subprocess line-queue fix into a different server architecture without an equivalent failing behavior.

**Port / issue.** **S** tests, **M** only for uncovered behavior; PRD #1 stories 13–18, 21–38. No new implementation issue was opened.

**Register.** `verified-in-source`: API metadata. `vendor claim`: PR test/performance results. `hypothesis`: local incremental-read benefit.

### F12 — The published speed is speculative output throughput

**Mechanism.** Strata's benchmark counts generated output tokens after speculative acceptance, rather than one target forward per output token. The denominator needed to explain the engine gap is therefore time per accepted output token, alongside time per target window and the number of distinct CPU misses in that window.

**Evidence.** [S0 `bench/results/2026-09-24-final/matrix.json:1–89`](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/bench/results/2026-09-24-final/matrix.json#L1-L89) reports Q2_0 at 128,478 input tokens with 65.09 generated tokens/s, 256 generated tokens, `spec_accept=0.687`, and `tokens_per_round=2.65`; the 3,562-token row reports 94.59 tokens/s and 3.23 tokens/round. [The serving loop:2155–2223](https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/src/program/generate.cpp#L2155-L2223) emits the accepted prefix plus target continuation and includes drafting/commit in the decode interval.

**EXL3 comparison.** X0's generation/verification uses the normal forward seam ([model.py:384–409](https://github.com/xenodeve/exllamav3-xeno/blob/12414d0af7b3beeabdda5990f6b554b996fa1416/exllamav3/model/model.py#L384-L409)); H0 says local MTP is presently a net loss. H0 does not completely specify the sampler, accepted-token denominator and exact placement for every 10–14 tokens/s observation, so an engine-only speedup ratio is not established.

**Port / issue.** **S**, Flash-Next #4 and PRD #1 measurement stories. Add denominators to the existing result register; no new benchmark campaign is run by this research session.

**Register.** `vendor claim`: throughput. `verified-in-source`: acceptance and emission accounting. `hypothesis`: achievable output throughput on our hardware.

## 3. What this research changes in the implementation queue

Keep the existing three-tier design and the implemented AVX-VNNI work. The genuinely useful additions are an enclosing graph/dispatch experiment, a fractional GPU-miss execution experiment at verification-sized batches, and an isolated investigation of residual mul1 reconstruction cost.

Do not spend another task merely proving that EXL3 has CPU offload, that the second GPU computes its resident experts, that PLE can stream, or that router skew exists. Those are already inputs. Also do not equate “RAM traffic is low” with “every kernel is host-bound”: a compute-heavy decoder and a waiting host can both produce low traffic. The measurements must separate them.

A minimal shared timing record is:

```text
source_revision, artifact_revision, local_patch_id, device_placement
actual_input_tokens, window_T, emitted_tokens, accepted_drafts
host_submit_ms, gpu_exposed_wait_ms, cpu_pool_ms
cpu_reconstruct_ms, cpu_dot_ms, cpu_epilogue_ms
cpu_distinct_experts, cpu_assignment_entries, gpu_miss_experts
h2d_weight_bytes, d2h_weight_bytes, cross_gpu_activation_bytes
verifier_ms, draft_ms, commit_ms, wall_decode_ms
ple_submit_ms, ple_wait_ms, available_ram, pinned_bytes
```

Timers around asynchronous launches are not kernel execution times. CUDA events spanning a GPU wait include that wait. CPU/GPU overlapped intervals cannot simply be added. These definitions are needed to answer this specific research question, not a replacement for the project's existing ABBA discipline.

The older 480 microseconds/layer observation corresponds arithmetically to about 23 ms over 48 layers. Because H0 says a direct fused call already reduced it, **23 ms is neither a newly discovered saving nor the current removable budget**. Similarly, the 4 ms/layer sample at 82% CPU capacity cannot be multiplied by the later 12% selection rate to predict a new runtime.

## 4. Ranked experiments most likely to reduce the remaining gap

### 1. Remove the remaining outer dispatch/submission bubbles

**Why first:** Strata has an executable whole-chain graph, whereas X0 exposes a Python layer loop; H0 independently reports costly remote dispatch. This is the strongest directly relevant mechanism to investigate, but the post-fused-call local cost still needs quantification.

**Experiment:** the building session runs its current baseline and a frozen-placement capture/native-dispatch prototype on the same token-ID trace, first at T=1 and then T=4. Start with one representative layer segment, keep arithmetic and ownership unchanged, and extend to a full target window only if host self-time or submission gaps actually shrink. Keep the established 4–6-P-core configuration rather than mixing in a thread sweep.

**Confirming observation:** fewer host submissions and smaller GPU idle gaps produce lower wall time per identical target window, with unchanged outputs and CPU expert workload. **Refuting observation:** host time falls but window time does not, because the exposed CPU reconstruction wait dominates.

**Cost / owner:** L; Flash-Next #5/#6, engine #2/#3. No numerical speedup promised.

### 2. Let the 4070 SUPER compute an economical share of verify-time CPU misses

**Why second:** Strata explicitly publishes GPU work before the CPU drain and can move a share of RAM misses to the GPU even during verification. This directly addresses H0's finding that MTP becomes expensive when the CPU sees a larger union of experts.

**Experiment:** preserve the existing persistent 5060 Ti slice, quantization, and ownership set; compare CPU-only misses with bounded GPU fractions, such as 0, 1/4, 1/2 and all eligible misses, on the 4070's transient staging path. Test T=1 and T=4 with the same routed trace before testing freely generated MTP output. This is an execution-policy ablation, not a simultaneous change to exclusive migration or cache size.

**Confirming observation:** CPU exposed wait and time per accepted output token fall enough to cover transfer and remote-combine costs. **Refuting observation:** PCIe/launch/combine time replaces rather than hides CPU time, or acceptance gains are insufficient. The winner may differ between T=1 and T=4.

**Cost / owner:** M–L; Flash-Next #5/#6/#9, engine #2/#3/#6.

### 3. Reduce the residual mul1 reconstruction bottleneck, using the current VNNI tier

**Why third:** Strata's Q2_0 decoder avoids the trellis/codebook reconstruction that H0 identifies as a likely compute limit. The local VNNI result already constrains expectations: the next step must target the remaining work rather than counting the same dot-product optimization twice.

**Experiment:** extract current gate/up and down calls using fixed activations and routed expert IDs; compare the existing AVX2 and implemented AVX-VNNI paths with a reconstruction-focused candidate. Attribute cycles to state extraction, integer reconstruction, dot accumulation and epilogue; inspect the actual generated code and register spills. Keep compressed weights and the existing parity contract. Then replay the realistic low-miss trace with row-task granularity varied independently.

**Confirming observation:** exact expert outputs with materially lower reconstruction time, followed by reduced exposed CPU wait in the same three-tier workload. **Refuting observation:** microkernel gains do not survive the full pipeline, or state extraction/register pressure cancels the proposed improvement.

**Cost / owner:** M; Flash-Next #7/#8, engine #4/#5. This is not a new request to implement AVX-VNNI, which the local session has already done.

**Decision:** pursue these three measurements on EXL3-xeno; retain Strata as an implementation reference and separately measured competitor. The source supports useful mechanisms, not a claim that porting them will necessarily turn 10–14 into 60 tokens/s.
