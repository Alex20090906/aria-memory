# Fast-weight memory patch for the Strata engine

Upstream: [Strata](https://github.com/Niko1221/Strata), version 0.1.52, commit `a42cc3d`, MIT licence
([LICENSE.Strata](LICENSE.Strata)). The files in this folder contain Strata's own lines and are distributed under
Strata's MIT licence, not under this repository's Apache-2.0.

| File | What |
|---|---|
| `fastmem.patch` | the complete change against `a42cc3d`: 10 files, 707 lines added, 4 removed. Applies with `git apply` on a clean checkout of `a42cc3d`; the result is identical to the research engine's last state. |
| `fmem.cu`, `fmem.hpp` | the kernel and its header after the patch (also inside the patch), for reading |

## What the patch adds

A sparse fast-weight memory that runs after a layer's residual write in every path (decode, speculative verify,
prefill):

- **apply:** each token's adapted hidden state is scored against D units, the top k rows of B are added back;
- **capture:** the units each token used are appended to a capture file, so writes can be computed in Python;
- **head mode** (`STRATA_FMEM_HEAD`): values are added just before the output layer, keys come from a key layer;
- **anchor mode** (`STRATA_FMEM_ANCHOR`): the memory is read only for the tokens after an anchor `[thing|aspect] ->`,
  the anchor is located by its token id and answer keys can leave out the previous token (`STRATA_FMEM_SHIFT_NOTOK`);
- **per-unit read weights** (anchor file v3), a **per-row scale vector** for decay and sleep shrinking without
  rewriting the memory file (`STRATA_FMEM_ROWSCALE`), and **dirty-row reload** of changed rows only;
- a `[[RAW]]` plain-text request mode in `serve/server.py` used by the research scripts;
- debug outputs (`STRATA_FMEM_DEBUG`, `STRATA_FMEM_DEBUG_TOK`) that write to stderr only, because the engine's
  stdout is its control pipe.

With no memory file (`STRATA_FMEM_FILE` unset) no layer is covered and the engine behaves as upstream.

## History

The patch squashes these research commits (2026-11-04 to 2026-11-05):

| Commit | Date | Change |
|---|---|---|
| e34330f | 11-04 12:42 | head mode |
| 4484ef9 | 11-04 13:02 | bias |
| b4abe40 | 11-04 17:01 | anchor mode |
| 3335088 | 11-04 17:12 | anchor debug prints (host side) |
| ef4f4a7 | 11-04 17:28 | mapped pos0, debug rows with B norm |
| 38ceb7c | 11-04 17:42 | per-row positions, logit sentinel |
| c5c618a | 11-04 17:57 | logit sentinels (debug 3/4/5) |
| 43aaa02 | 11-04 18:34 | anchor located by token id (STRFANC2) |
| 762c5e5 | 11-04 22:13 | relative anchor position and answer keys without previous token |
| e95a6fa | 11-05 00:50 | per-unit read weights (anchor v3) |
| a908f05 | 11-05 11:32 | per-row scale vector and dirty-row reload |
| 818a934 | 11-05 11:33 | serve: `[[RAW]]` plain-text requests |

The first local commit also contained an early draft of `fmem.cu` and `fmem.hpp`; the patch is computed against the
upstream files, so it includes them in full.

## Using it with the research scripts

The Python side (writes, anchors, sleep, the weakest-token confidence threshold, the echo signature) is in
`experiments/llm_fastmem/` (`fmem_tool.py`, `fmem_anchor.py`, `sleep.py`, `memsvc.py`). In the public copy of those
scripts the engine's name was replaced by `llm`, so their environment variables read `LLM_FMEM_*` where the engine
reads `STRATA_FMEM_*`. Map the names back before running them against a patched engine. Building the engine needs
CUDA and a GPU (the research built it for sm_86 with CUDA 13.0 and Visual Studio 2022); see Strata's own build
instructions.
