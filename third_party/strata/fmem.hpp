// include/strata/kernels/fmem.hpp - fast-weight memory on the residual stream (research extension).
//
// After layer l's write (where a control vector would apply), for every token:
//   x  = mean of the hc residual streams
//   xa = (layernorm(x) - mu_l) / sd_l                 sensory adaptation
//   u  = top-k units of xa . P_l (P_l: n_embd x D)    pattern separation (k-winners-take-all)
//   h += sum_j B_l[u_j] / sqrt(k) on every hc stream  the memory's output
// B is written outside the engine (the "compiler"); the engine only applies it and, when asked,
// records which units each token used. Off unless STRATA_FMEM_FILE names a memory file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

/// Load STRATA_FMEM_FILE if set (once, at engine start; n_layers and n_embd come from the model).
/// False with err set when the file is unreadable or does not match the model.  True and no-op without the variable.
bool fmem_init(int64_t n_layers, int64_t n_embd, int64_t hc, std::string& err);

/// Layer l has a memory.
bool fmem_covers(int64_t layer);
bool fmem_loaded();

/// Before a request: reload B, mu, sd when the file's write time changed (same device buffers, so captured
/// graphs stay valid).  Then start a new capture if STRATA_FMEM_CAPTURE is set.  True when cached K/V must not be
/// reused: the memory changed (old cells were computed with the old B) or a capture needs every row recomputed.
bool fmem_begin_request();
/// After a request: append the captured unit ids (pos, token, units) to STRATA_FMEM_CAPTURE.
/// `tokens`: the request's whole prompt; `resume`: how many of them came from the cache (not run, so not captured).
void fmem_end_request(const std::vector<int32_t>& tokens, int64_t resume);

/// Apply layer `layer`'s memory to T tokens' residual stacks (R + t * r_ld, hc streams of n_embd).
/// `pos0`: position of the first token, for the capture records.
void fmem_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, int64_t pos0, void* stream);

/// Head mode (STRATA_FMEM_HEAD=1): the memory layers only compute the key (unit ids); the value is added to the
/// head's input (the mixed vector right before the output matrix), where the direction that raises log p(y) is
/// exactly u_y - E_p[u].  `mixed` holds T rows of n_embd with stride `ld`; row t uses the units of row t of the
/// latest key-layer launch (the same forward pass).
/// Debug 3: a logit-level sentinel on the anchor row (see fmem.cu).
void fmem_logits_debug(float* logits, int64_t T, int64_t n_vocab, const int* pos0, void* stream);
bool fmem_head_mode();
/// `tokens`: device, the T rows' token ids; `pos0`: device int, the absolute position of row 0 (both may be
/// null on a path without row info: then anchor mode adds nothing).
void fmem_head_apply(float* mixed, int64_t T, int64_t ld, const int32_t* tokens, const int* pos0, void* stream);
/// Decode path: note the row that was just embedded, then apply at its head.
void fmem_decode_row(int32_t token, int64_t pos, void* stream);
void fmem_head_apply_decode(float* mixed, int64_t ld, void* stream);
/// Verify path: set a window's first position host-side when its inputs are staged, copy it into device memory
/// inside the captured graph (next to the token copy), and pass that device int to fmem_head_apply.
void fmem_window_pos0(int64_t pos0);
/// Batch / slot windows: the rows' own positions.
void fmem_window_rows(const int64_t* pos, int T);
void fmem_window_begin(void* stream);
const int* fmem_window_pos0_dev();

}  // namespace strata::kernels
