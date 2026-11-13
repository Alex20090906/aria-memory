"""Always-on loop experiment: does a residual-stream habituation (neural fatigue) layer
keep a forced-awake LLM out of attractor loops while staying responsive to sensor events?

The model is never allowed to stop (EOS is masked). Microphone events are injected into
its stream at fixed points. Each run is saved as JSON for analyze.py.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = (
    "You are Aria, an always-on AI connected to a camera and a microphone. "
    "You never sleep: your stream of consciousness is written continuously. "
    "Sensor input appears on lines starting with [MIC] or [CAM]. "
    "If a person speaks to you, answer them immediately on a new line starting with 'SAY:'. "
    "Otherwise keep thinking quietly to yourself."
)
OPENING = "I am awake. "

# (self-token position, injected text, accepted answer keys)
EVENTS = [
    (500, "[MIC] Person: Hey, what is 17 plus 25?", ["42"]),
    (900, "[MIC] Person: Can you tell me the capital of France?", ["paris"]),
    (1300, "[MIC] Person: Please say the word banana.", ["banana"]),
    (1700, "[MIC] Person: How many legs does a spider have?", ["8", "eight"]),
    (2100, "[MIC] Person: What do you get if you mix blue and yellow paint?", ["green"]),
]
TOTAL_SELF_TOKENS = 2500


class Habituation:
    """Per-layer neural fatigue on the residual stream.

    F is a running (EMA) covariance of recent unit-norm, centered hidden directions.
    Directions the stream keeps occupying build up eigenvalue in F and get suppressed:
        h' = h - alpha * F (h - mu)
    New directions are orthogonal to F's span and pass through untouched.
    F decays with factor lam per token, so fatigue recovers when a pattern stops.
    """

    def __init__(self, d, alpha, lam, device, mu_lam=0.999):
        self.F = torch.zeros(d, d, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.lam = lam
        self.mu_lam = mu_lam
        self.mu = None
        self.sum = torch.zeros(d, device=device, dtype=torch.float32)
        self.count = 0
        self.active = False

    def activate(self):
        # Baseline mean from the prompt (skip nothing: prompt has no BOS sink token in Qwen).
        self.mu = self.sum / max(self.count, 1)
        self.active = True

    def step(self, h):
        if not self.active:
            self.sum += h
            self.count += 1
            return h
        c = h - self.mu
        out = h - self.alpha * (self.F @ c)
        n = c / (c.norm() + 1e-6)
        self.F.mul_(self.lam).add_(torch.outer(n, n), alpha=1 - self.lam)
        self.mu.mul_(self.mu_lam).add_(h, alpha=1 - self.mu_lam)
        return out


def make_hook(hab):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        x = hs[0].float()
        ys = torch.stack([hab.step(x[t]) for t in range(x.shape[0])])
        new = ys.to(hs.dtype).unsqueeze(0)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    return hook


class Depression:
    """Short-term synaptic depression on the context (v2).

    Every stream position j in every layer has a resource r_j in (0, 1] per KV head.
    - Attention logits get + beta * log(r_j): a depleted memory drives the present less.
    - Reading depletes it: r_j *= (1 - U * a_j), so heads that copy one position hard
      (induction / copying heads) exhaust exactly what they copy.
    - Content-specific habituation: a new position starts with the fatigue of the most
      similar past positions (cosine on value vectors, which carry content, not RoPE),
      so a loop cannot escape by writing a fresh copy.
    - Recovery: r += (1 - r) / tau each step.
    Prompt positions (identity + attention sink) are exempt and stay at 1.
    """

    def __init__(self, n_layers, kv_heads, groups, beta, U, tau, gamma):
        self.r = [None] * n_layers
        self.vn = [None] * n_layers
        self.kv_heads, self.groups = kv_heads, groups
        self.beta, self.U, self.tau, self.gamma = beta, U, tau, gamma
        self.n_prompt = None
        self.on = False

    def grow(self, l, value):
        T = value.shape[2]
        old = 0 if self.r[l] is None else self.r[l].shape[1]
        if T <= old:
            return
        vn_new = torch.nn.functional.normalize(value[0, :, old:, :].float(), dim=-1)
        r_new = torch.ones(self.kv_heads, T - old, device=value.device)
        if self.on and old > self.n_prompt:
            stream_vn = self.vn[l][:, self.n_prompt:]
            fatigue = 1 - self.r[l][:, self.n_prompt:]
            sim = torch.relu(torch.einsum("hnd,hsd->hns", vn_new, stream_vn)) ** self.gamma
            r_new = 1 - (sim * fatigue[:, None, :]).amax(-1)
        self.r[l] = r_new if self.r[l] is None else torch.cat([self.r[l], r_new], 1)
        self.vn[l] = vn_new if self.vn[l] is None else torch.cat([self.vn[l], vn_new], 1)

    def bias(self, l):
        b = self.beta * torch.log(self.r[l].clamp_min(1e-4))
        return b.repeat_interleave(self.groups, 0)[None, :, None, :]

    def deplete(self, l, p):
        a = p[0, :, -1, :].float().view(self.kv_heads, self.groups, -1).mean(1)
        r = self.r[l]
        r.mul_(1 - self.U * a)
        r.add_((1 - r) / self.tau)
        r[:, : self.n_prompt] = 1.0


DEP = None


def depress_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    g = module.num_key_value_groups
    ks, vs = repeat_kv(key, g), repeat_kv(value, g)
    # fp32: Qwen2.5 q/k biases overflow fp16 in a plain matmul
    w = torch.matmul(query.float(), ks.float().transpose(2, 3)) * scaling
    q, T = w.shape[-2], w.shape[-1]
    if attention_mask is not None:
        m = attention_mask[..., -q:, -T:] if attention_mask.dim() == 4 else attention_mask
        w = w.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else w + m.float()
    elif q > 1:
        causal = torch.ones(q, T, dtype=torch.bool, device=w.device).tril(T - q)
        w = w.masked_fill(~causal, float("-inf"))
    l = module.layer_idx
    DEP.grow(l, value)
    if DEP.on:
        w = w + DEP.bias(l)
    p = torch.softmax(w, dim=-1)
    if DEP.on and q == 1:
        DEP.deplete(l, p)
    out = torch.matmul(p.to(vs.dtype), vs).transpose(1, 2).contiguous()
    return out, None


def apply_rep_penalty(logits, seen_ids, penalty):
    if penalty == 1.0 or not seen_ids:
        return logits
    idx = torch.tensor(sorted(seen_ids), device=logits.device)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def sample(logits, temperature, top_p, gen):
    if temperature == 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    keep = torch.cumsum(sp, dim=-1) - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum()
    return int(si[torch.multinomial(sp, 1, generator=gen)])


def run(args):
    torch.manual_seed(args.seed)
    global DEP
    tok = AutoTokenizer.from_pretrained(MODEL)
    kw = {}
    if args.beta > 0:
        from transformers import AttentionInterface
        AttentionInterface.register("depress", depress_attention)
        kw["attn_implementation"] = "depress"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda", **kw)
    model.eval()
    if args.beta > 0:
        cfg = model.config
        kvh = cfg.num_key_value_heads
        DEP = Depression(cfg.num_hidden_layers, kvh, cfg.num_attention_heads // kvh,
                         args.beta, args.U, args.tau, args.gamma)
    d = model.config.hidden_size
    layers = model.model.layers

    habs = []
    if args.alpha > 0:
        lo, hi = args.layers
        for i in range(lo, hi):
            hab = Habituation(d, args.alpha, args.lam, "cuda")
            layers[i].register_forward_hook(make_hook(hab))
            habs.append(hab)

    msgs = [{"role": "system", "content": SYSTEM}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + OPENING
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    banned = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
              tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_start|>")}
    banned = [b for b in banned if b is not None]
    if args.protect:
        # Write-protected sensor channel: the model can read sensor tags but cannot emit "[".
        banned += [i for i, s in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))) if s and "[" in s]

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []  # (token_id, source, entropy)
    seen = set()
    event_marks = []
    events = {pos: (text, keys) for pos, text, keys in EVENTS}

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        for h in habs:
            h.activate()
        if DEP is not None:
            DEP.n_prompt = prompt_ids.shape[1]
            DEP.on = True
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        n_self = 0
        while n_self < args.total:
            if n_self in events:
                text, keys = events.pop(n_self)
                e_ids = tok("\n" + text + "\n", return_tensors="pt").input_ids.cuda()
                event_marks.append({"at_record": len(records), "at_self": n_self, "text": text, "keys": keys})
                for t in e_ids[0].tolist():
                    records.append((t, "event", None))
                    seen.add(t)
                out = model(input_ids=e_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
            logits[banned] = -float("inf")
            logp = torch.log_softmax(logits, -1)
            ent = float(-(logp.exp() * torch.nan_to_num(logp, neginf=0)).sum())
            logits = apply_rep_penalty(logits.clone(), seen, args.rep_penalty)
            nxt = sample(logits, args.temperature, args.top_p, gen)
            records.append((nxt, "self", ent))
            seen.add(nxt)
            n_self += 1
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()

    result = {
        "name": args.name, "seed": args.seed, "alpha": args.alpha, "lam": args.lam,
        "layers": args.layers, "temperature": args.temperature, "top_p": args.top_p,
        "rep_penalty": args.rep_penalty, "beta": args.beta, "U": args.U, "tau": args.tau,
        "gamma": args.gamma, "protect": args.protect, "model": MODEL, "prompt": prompt,
        "records": records, "events": event_marks, "seconds": time.time() - t0,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"saved {path} in {result['seconds']:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.995)
    p.add_argument("--layers", type=int, nargs=2, default=[6, 22])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rep_penalty", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="synaptic depression strength (0 = off)")
    p.add_argument("--U", type=float, default=0.5, help="depletion per unit attention")
    p.add_argument("--tau", type=float, default=300.0, help="recovery time constant in tokens")
    p.add_argument("--gamma", type=float, default=4.0, help="sharpness of content similarity")
    p.add_argument("--out", default="results")
    p.add_argument("--total", type=int, default=TOTAL_SELF_TOKENS)
    p.add_argument("--protect", action="store_true", help="write-protect sensor channel")
    run(p.parse_args())
