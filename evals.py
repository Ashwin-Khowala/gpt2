"""
evals.py - evaluation suite for checkpoints trained with train_gpt2.py

What it measures (every number comes with a sample size or standard error where it makes sense):
  sanity     param count, NaN/inf weights, weight tying, weight statistics
  val        held-out loss / perplexity / bits-per-byte on the FineWeb-Edu val shard (+ std. error)
  domains    the same loss on YOUR OWN text files (code, chat, wiki, ...) -> out-of-domain behaviour
  hellaswag  your own implementation (same as training): acc and acc_norm +- binomial std. error
  lm_eval    EleutherAI lm-evaluation-harness (hellaswag, arc, piqa, winogrande, lambada, ...)
  generate   fixed-seed samples + a repetition score (distinct-3)

Results go to results/<name>.json, and results/summary.md + summary.csv compare every run side by side.

Examples
  # coarse health check of a checkpoint (CPU-friendly, minutes)
  CUDA_VISIBLE_DEVICES="" python evals.py --ckpt log/model_20000.pt --suite quick

  # full evaluation of the final model, with OpenAI GPT-2 124M as a yardstick (use a free GPU)
  python evals.py --ckpt log/model_final.pt --baseline --suite full --domain code=samples/code.txt

  # compare many checkpoints, then plot the training log
  python evals.py --ckpt log/model_05000.pt log/model_10000.pt log/model_15000.pt --suite quick
  python evals.py --plot_log log/log.txt --tokens_per_step 32768

  # any HuggingFace model through lm-eval only (e.g. a stronger reference)
  python evals.py --hf_model EleutherAI/pythia-160m --only lm_eval --lm_limit 500
"""
import os
import sys
import json
import math
import time
import argparse
import glob
import csv
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig

TRANSPOSED = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

# metric shown in the summary table for each lm-eval task
PRIMARY = {
    "hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "piqa": "acc",
    "winogrande": "acc", "lambada_openai": "acc", "openbookqa": "acc_norm",
}
DEFAULT_LM_TASKS = ",".join(PRIMARY)
DEFAULT_PROMPTS = ["Hello, I'm a language model,", "The capital of France is", "In 1969, the first humans"]

# CPU-friendly presets. A 124M model costs ~0.25 GFLOP per token, so 1M tokens on a small CPU is slow.
SUITES = {
    "quick": dict(tests="sanity,val,hellaswag,generate", val_tokens=300_000, hella_limit=300, lm_limit=200),
    "full":  dict(tests="sanity,val,domains,hellaswag,lm_eval,generate", val_tokens=0, hella_limit=0, lm_limit=0),
}


# ------------------------------- device / loading ----------------------------
def pick_device(choice, force):
    """Never grab a GPU that a training run is using: fall back to CPU if little memory is free."""
    if choice == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    free, _ = torch.cuda.mem_get_info()
    if free < 4e9 and not force:
        print(f"[device] only {free/1e9:.1f} GB of GPU memory is free (is training running?) -> using CPU. "
              f"Pass --force to use the GPU anyway.")
        return torch.device("cpu")
    return torch.device("cuda")


def load_ours(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        print("  weights_only load failed; falling back to full unpickle (only do this for files you trust)")
        ck = torch.load(path, map_location="cpu", weights_only=False)  # old checkpoints pickled the GPTConfig object
    cfg = ck["config"]
    if not isinstance(cfg, GPTConfig):
        cfg = GPTConfig(**cfg)
    model = GPT(cfg)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)  # strict: a mismatch is an error, never a silent skip
    return model, {"step": ck.get("step"), "ckpt_val_loss": ck.get("val_loss")}


def load_openai_gpt2():
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    model = GPT(GPTConfig())  # vocab 50257
    sd, sd_hf = model.state_dict(), hf.state_dict()
    with torch.no_grad():
        for k in sd:
            sd[k].copy_(sd_hf[k].t() if any(k.endswith(w) for w in TRANSPOSED) else sd_hf[k])
    return model, hf


def to_hf(model):
    """GPT-2 architecture == HuggingFace GPT2LMHeadModel (Conv1D weights are transposed)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    c = model.config
    hf = GPT2LMHeadModel(GPT2Config(
        vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd, n_layer=c.n_layer, n_head=c.n_head,
        activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5))
    sd, sd_hf = model.state_dict(), hf.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            assert k in sd_hf, f"key {k} missing in the HF model"
            sd_hf[k].copy_(v.t() if any(k.endswith(w) for w in TRANSPOSED) else v)
    return hf.eval()


def parity_check(model, hf, device):
    """Our model and the converted HF model must give the same logits, otherwise lm-eval numbers are meaningless."""
    ids = torch.randint(0, model.config.vocab_size, (1, 64))
    with torch.no_grad():
        a = model(ids.to(next(model.parameters()).device))[0].float().cpu()
        b = hf(ids.to(next(hf.parameters()).device)).logits.float().cpu()
    diff = (a - b).abs().max().item()
    print(f"  HF export parity: max |logit diff| = {diff:.2e}")
    assert diff < 1e-3, "HF conversion does not match the original model - refusing to run lm-eval"
    return diff


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"))


# ------------------------------- data ----------------------------------------
def token_bytes_table(enc):
    """UTF-8 byte length of every token id, used for tokenizer-independent bits-per-byte."""
    t = np.zeros(enc.n_vocab, dtype=np.int64)
    for i in range(enc.n_vocab):
        try:
            t[i] = len(enc.decode_single_token_bytes(i))
        except Exception:
            t[i] = 0
    t[enc.eot_token] = 0  # <|endoftext|> is a separator, not text
    return t


def load_val_tokens(data_root):
    shards = sorted(f for f in os.listdir(data_root) if "val" in f)
    assert shards, f"no val shard found in {data_root}"
    return np.load(os.path.join(data_root, shards[0]), mmap_mode="r")


def load_domain(path, enc):
    if path.endswith(".npy"):
        return np.load(path).astype(np.int32)
    text = open(path, encoding="utf-8", errors="ignore").read()
    return np.array([enc.eot_token] + enc.encode_ordinary(text), dtype=np.int32)


# ------------------------------- metrics -------------------------------------
def sanity(model):
    c = model.config
    n = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    named = dict(model.named_parameters())
    bad = [k for k, p in named.items() if not torch.isfinite(p).all()]
    tied = model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    w = named["transformer.wte.weight"]
    out = {"params_M": round(n / 1e6, 2), "non_finite_tensors": bad, "wte_lm_head_tied": bool(tied),
           "wte_std": round(w.std().item(), 4), "ln_vocab": round(math.log(c.vocab_size), 3)}
    print(f"  sanity: {out['params_M']}M params, non-finite: {bad or 'none'}, tied: {tied}, "
          f"(a model that knows nothing has loss ~{out['ln_vocab']})")
    return out


@torch.no_grad()
def window_loss(model, tokens, device, B, T, max_tokens, tok_bytes, label=""):
    """Loss on non-overlapping T-token windows. Reports loss, perplexity, bits-per-byte and std. error."""
    n_avail = (len(tokens) - 1) // T
    n_win = n_avail if not max_tokens else min(n_avail, max_tokens // T)
    assert n_win > 0, f"not enough tokens for even one window of {T}"
    nll, ntok, nbytes, wl = 0.0, 0, 0, []
    for i in range(0, n_win, B):
        b = min(B, n_win - i)
        starts = [(i + j) * T for j in range(b)]
        x_np = np.stack([np.asarray(tokens[s:s + T]) for s in starts]).astype(np.int64)
        y_np = np.stack([np.asarray(tokens[s + 1:s + T + 1]) for s in starts]).astype(np.int64)
        x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        with amp(device):
            logits, _ = model(x)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none").view(b, T)
        wl.extend(ce.mean(1).tolist())
        nll += ce.sum().item()
        ntok += b * T
        nbytes += int(tok_bytes[y_np].sum())
    loss = nll / ntok
    se = float(np.std(wl, ddof=1) / math.sqrt(len(wl))) if len(wl) > 1 else float("nan")
    out = {"loss": round(loss, 4), "se": round(se, 4), "ppl": round(math.exp(loss), 2),
           "bpb": round(nll / (math.log(2) * max(nbytes, 1)), 4), "tokens": ntok}
    print(f"  {label or 'val'}: loss {out['loss']} +- {out['se']} | ppl {out['ppl']} | bits/byte {out['bpb']} | {ntok/1e6:.2f}M tokens")
    return out


@torch.no_grad()
def eval_hellaswag(model, device, limit):
    from hellaswag import render_example, iterate_examples  # your existing hellaswag.py
    c = cn = n = 0
    for ex in iterate_examples("val"):
        _, tokens, mask, label = render_example(ex)
        tokens, mask = tokens.to(device), mask.to(device)
        with amp(device):
            logits, _ = model(tokens)
        logits = logits.float()
        sl = logits[..., :-1, :].contiguous()
        st = tokens[..., 1:].contiguous()
        losses = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), reduction="none").view(tokens.size(0), -1)
        sm = mask[..., 1:].contiguous()
        sum_loss = (losses * sm).sum(1)
        avg_loss = sum_loss / sm.sum(1)
        c += int(sum_loss.argmin().item() == label)    # acc: total log-likelihood
        cn += int(avg_loss.argmin().item() == label)   # acc_norm: per-token average (the one people report)
        n += 1
        if n % 1000 == 0:
            print(f"    hellaswag {n} examples, acc_norm so far {cn/n:.4f}")
        if limit and n >= limit:
            break
    p = cn / n
    out = {"acc": round(c / n, 4), "acc_norm": round(p, 4), "se_norm": round(math.sqrt(p * (1 - p) / n), 4), "n": n}
    print(f"  hellaswag: acc_norm {out['acc_norm']} +- {out['se_norm']} (n={n}, chance 0.25) | acc {out['acc']}")
    return out


def run_lm_eval(hf_model, device, tasks, limit, batch):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    hf_model = hf_model.to(device).eval()
    lm = HFLM(pretrained=hf_model, tokenizer=AutoTokenizer.from_pretrained("gpt2"),
              batch_size=batch, device=str(device))
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=(limit or None))
    out = {}
    for task, metrics in res["results"].items():
        out[task] = {k.split(",")[0]: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        prim = PRIMARY.get(task)
        if prim in out[task]:
            se = out[task].get(prim + "_stderr")
            print(f"  lm-eval {task}: {prim} {out[task][prim]:.4f}" + (f" +- {se:.4f}" if se else ""))
    return out


@torch.no_grad()
def generate(model, enc, device, prompts, max_new=64, top_k=50, temp=0.8, n=3, seed=42):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    results = []
    for prompt in prompts:
        x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1)
        for _ in range(max_new):
            with amp(device):
                logits, _ = model(x[:, -model.config.block_size:])
            probs = F.softmax(logits[:, -1, :].float() / temp, dim=-1).cpu()
            tp, ti = torch.topk(probs, top_k, dim=-1)
            nxt = torch.gather(ti, -1, torch.multinomial(tp, 1, generator=g)).to(device)
            x = torch.cat((x, nxt), dim=1)
        samples = [enc.decode(r.tolist()) for r in x]
        # distinct-3: share of unique token trigrams; low values mean the model loops/repeats itself
        d3 = []
        for r in x.tolist():
            tri = list(zip(r, r[1:], r[2:]))
            d3.append(len(set(tri)) / max(len(tri), 1))
        results.append({"prompt": prompt, "samples": samples, "distinct3": round(float(np.mean(d3)), 3)})
        print(f"  generate: {prompt!r} -> distinct-3 {results[-1]['distinct3']}")
        print(f"    e.g. {samples[0][:200]!r}")
    return results


# ------------------------------- orchestration -------------------------------
def evaluate(name, model, hf_model, meta, args, tests, device, enc, tok_bytes, val_tokens, domains, is_baseline):
    print(f"\n########## {name} ##########")
    res = {"name": name, "suite": args.suite, "is_baseline": is_baseline, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "device": str(device), **meta}
    if meta.get("step") is not None:
        res["tokens_seen_M"] = round(meta["step"] * args.tokens_per_step / 1e6, 1)
    if model is not None:
        model.to(device).eval()
    if "sanity" in tests and model is not None:
        res["sanity"] = sanity(model)
    if "val" in tests and model is not None:
        res["val"] = window_loss(model, val_tokens, device, args.B, args.T, args.val_tokens, tok_bytes, "val (FineWeb-Edu)")
    if "domains" in tests and model is not None and domains:
        res["domains"] = {}
        for dn, dt in domains.items():
            res["domains"][dn] = window_loss(model, dt, device, args.B, args.T, args.val_tokens, tok_bytes, f"domain:{dn}")
    if "hellaswag" in tests and model is not None:
        res["hellaswag"] = eval_hellaswag(model, device, args.hella_limit)
    if "lm_eval" in tests:
        if hf_model is None:
            hf_model = to_hf(model)
            parity_check(model, hf_model, device)
        res["lm_eval"] = run_lm_eval(hf_model, device, args.lm_tasks.split(","), args.lm_limit, args.lm_batch)
        res["lm_eval_limit"] = args.lm_limit
    if "generate" in tests and model is not None:
        res["generation"] = generate(model, enc, device, args.prompt or DEFAULT_PROMPTS)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{name.replace('/', '_')}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  -> saved {path}")
    return res


def fmt(x, d=4):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def summarize(out_dir):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))]
    if not rows:
        return
    rows.sort(key=lambda r: (r.get("is_baseline", False), r.get("step") or 0, r["name"]))
    dnames = sorted({d for r in rows for d in r.get("domains", {})})
    tasks = [t for t in PRIMARY if any(t in r.get("lm_eval", {}) for r in rows)]
    header = (["model", "step", "tokens(M)", "val loss", "val bpb"] + [f"{d} loss" for d in dnames]
              + ["HellaSwag acc_norm (own)"] + [f"{t} {PRIMARY[t]}" for t in tasks])
    table = []
    for r in rows:
        v, h = r.get("val", {}), r.get("hellaswag", {})
        row = [r["name"], r.get("step"), r.get("tokens_seen_M"),
               f"{v['loss']:.4f}+-{v['se']:.4f}" if v else "-", fmt(v.get("bpb")) if v else "-"]
        row += [fmt(r.get("domains", {}).get(d, {}).get("loss")) for d in dnames]
        row += [f"{h['acc_norm']:.4f}+-{h['se_norm']:.4f} (n={h['n']})" if h else "-"]
        for t in tasks:
            m = r.get("lm_eval", {}).get(t, {})
            val = m.get(PRIMARY[t])
            se = m.get(PRIMARY[t] + "_stderr")
            row.append("-" if val is None else (f"{val:.4f}+-{se:.4f}" if se else f"{val:.4f}"))
        table.append(["-" if c is None else str(c) for c in row])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print("\n" + "\n".join(md) + f"\n\n(saved {out_dir}/summary.md and summary.csv)")


def plot_log(log_path, out_png, tokens_per_step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [l.split() for l in open(log_path) if l.strip()]

    def series(kind):
        a = np.array([(int(r[0]), float(r[2])) for r in rows if len(r) == 3 and r[1] == kind])
        return a if len(a) else None

    tr, va, hs = series("train"), series("val"), series("hella")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    if tr is not None:
        w = min(200, len(tr))
        sm = np.convolve(tr[:, 1], np.ones(w) / w, mode="valid")
        ax[0].plot(tr[w - 1:, 0] * tokens_per_step / 1e6, sm, label=f"train ({w}-step average)")
    if va is not None:
        ax[0].plot(va[:, 0] * tokens_per_step / 1e6, va[:, 1], label="val")
    ax[0].set_xlabel("tokens (M)")
    ax[0].set_ylabel("loss")
    ax[0].set_ylim(2.8, 6)
    ax[0].legend()
    ax[0].grid(alpha=.3)
    if hs is not None:
        ax[1].plot(hs[:, 0] * tokens_per_step / 1e6, hs[:, 1], marker="o")
    ax[1].axhline(0.25, ls="--", color="gray", label="chance")
    ax[1].set_xlabel("tokens (M)")
    ax[1].set_title("HellaSwag acc_norm")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=[], help="one or more checkpoints (model_XXXXX.pt, ckpt_latest.pt, model_final.pt)")
    ap.add_argument("--baseline", action="store_true", help="also evaluate OpenAI GPT-2 124M with identical tests")
    ap.add_argument("--hf_model", default=None, help="any HF model, evaluated with lm-eval only")
    ap.add_argument("--suite", choices=list(SUITES), default="quick")
    ap.add_argument("--only", default=None, help="comma list overriding the suite: sanity,val,domains,hellaswag,lm_eval,generate")
    ap.add_argument("--val_tokens", type=int, default=None, help="0 = whole val shard")
    ap.add_argument("--domain", action="append", default=[], help="name=path/to/file.txt (or .npy of token ids); repeatable")
    ap.add_argument("--hella_limit", type=int, default=None, help="0 = all ~10k examples")
    ap.add_argument("--lm_tasks", default=DEFAULT_LM_TASKS)
    ap.add_argument("--lm_limit", type=int, default=None, help="examples per lm-eval task, 0 = all")
    ap.add_argument("--lm_batch", type=int, default=8)
    ap.add_argument("--prompt", action="append", help="generation prompt, repeatable")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--force", action="store_true", help="use the GPU even when it has little free memory")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--data_root", default="edu_fineweb10B")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tokens_per_step", type=int, default=32768, help="your total_batch_size, to convert steps to tokens")
    ap.add_argument("--plot_log", default=None, help="path to log/log.txt -> writes <out>/curves.png")
    args = ap.parse_args()

    if args.plot_log:
        os.makedirs(args.out, exist_ok=True)
        plot_log(args.plot_log, os.path.join(args.out, "curves.png"), args.tokens_per_step)
        if not (args.ckpt or args.baseline or args.hf_model):
            return
    if not (args.ckpt or args.baseline or args.hf_model):
        summarize(args.out)
        return

    preset = SUITES[args.suite]
    tests = set((args.only or preset["tests"]).split(","))
    args.val_tokens = preset["val_tokens"] if args.val_tokens is None else args.val_tokens
    args.hella_limit = preset["hella_limit"] if args.hella_limit is None else args.hella_limit
    args.lm_limit = preset["lm_limit"] if args.lm_limit is None else args.lm_limit
    if args.domain and "domains" not in tests and not args.only:
        tests.add("domains")

    device = pick_device(args.device, args.force)
    print(f"device: {device} | suite: {args.suite} | tests: {sorted(tests)} | val_tokens: {args.val_tokens or 'all'} | "
          f"hella_limit: {args.hella_limit or 'all'} | lm_limit: {args.lm_limit or 'all'}")
    enc = tiktoken.get_encoding("gpt2")
    need_data = bool(tests & {"val", "domains"}) and (args.ckpt or args.baseline)
    tok_bytes = token_bytes_table(enc) if need_data else None
    val_tokens = load_val_tokens(args.data_root) if need_data and "val" in tests else None
    domains = {}
    for spec in args.domain:
        dn, path = spec.split("=", 1)
        domains[dn] = load_domain(path, enc)
        print(f"domain {dn}: {len(domains[dn])/1e6:.2f}M tokens from {path}")

    for ckpt in args.ckpt:
        model, meta = load_ours(ckpt)
        meta["ckpt"] = ckpt
        name = os.path.splitext(os.path.basename(ckpt))[0]
        evaluate(name, model, None, meta, args, tests, device, enc, tok_bytes, val_tokens, domains, False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.baseline:
        model, hf = load_openai_gpt2()
        evaluate("gpt2-openai-124M", model, hf, {"step": None}, args, tests, device, enc, tok_bytes, val_tokens, domains, True)
    if args.hf_model:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(args.hf_model)
        evaluate(args.hf_model, None, hf, {"step": None}, args, {"lm_eval"}, device, enc, None, None, {}, True)

    summarize(args.out)


if __name__ == "__main__":
    main()