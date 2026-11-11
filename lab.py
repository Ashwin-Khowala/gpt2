"""
lab.py - a playground for poking at your trained model.

Put it next to evals.py and model.py, then in a notebook:

    from lab import *
    load("log/model_final.pt")
    peek("The capital of France is")

Every function explains itself in its docstring. All of them use the model loaded by load().
"""
import io
import glob
import html
import contextlib
import torch
import torch.nn.functional as F
import tiktoken
from evals import load_ours, load_val_tokens, token_bytes_table, window_loss

enc = tiktoken.get_encoding("gpt2")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V = 50257          # real vocabulary size (the model's output is padded to 50304)
_model = None


def load(path="log/model_final.pt"):
    """Load a checkpoint. Every other function in this file uses it."""
    global _model
    _model, meta = load_ours(path)
    _model.to(dev).eval()
    print(f"loaded {path} (step {meta.get('step')}) on {dev}")
    return _model


def _m(m=None):
    m = m or _model
    assert m is not None, "call load('log/model_final.pt') first"
    return m


# ---------------------------------------------------------------------------
# 1. what does the model think comes next?
# ---------------------------------------------------------------------------
@torch.no_grad()
def peek(prompt, k=10):
    """Print the k most likely next tokens with their probabilities."""
    m = _m()
    x = torch.tensor([enc.encode(prompt)], device=dev)
    p = F.softmax(m(x)[0][0, -1, :V].float(), -1)
    top = p.topk(k)
    print(f"prompt: {prompt!r}")
    for pr, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {pr:6.1%}  {enc.decode([t])!r}")


# ---------------------------------------------------------------------------
# 2. where is the model surprised?
# ---------------------------------------------------------------------------
@torch.no_grad()
def token_losses(text, m=None):
    """Return (token_ids, loss_of_each_token_after_the_first) in nats."""
    m = _m(m)
    ids = enc.encode(text)[:m.config.block_size]
    assert len(ids) >= 2, "need at least 2 tokens"
    x = torch.tensor([ids], device=dev)
    lp = F.log_softmax(m(x)[0][0, :-1, :V].float(), -1)
    nll = (-lp.gather(1, x[0, 1:, None])[:, 0]).tolist()
    return ids, nll


def surprise(text):
    """Show text with a red background where the model was surprised (darker = more surprised).
    Hover a token in the notebook to see its loss."""
    from IPython.display import HTML, display
    ids, nll = token_losses(text)
    parts = [html.escape(enc.decode([ids[0]]))]
    for t, l in zip(ids[1:], nll):
        parts.append(f"<span style='background:rgba(255,70,70,{min(l / 10, 1):.2f})' title='{l:.1f} nats'>"
                     f"{html.escape(enc.decode([t]))}</span>")
    display(HTML("<div style='font-family:monospace;white-space:pre-wrap;line-height:1.6'>" + "".join(parts) + "</div>"))


def text_loss(text):
    """Average loss per token on a piece of text. Lower = the model finds this kind of text easier."""
    _, nll = token_losses(text)
    return sum(nll) / len(nll)


# ---------------------------------------------------------------------------
# 3. generating text, sampling lab
# ---------------------------------------------------------------------------
@torch.no_grad()
def complete(prompt, n=60, temp=0.8, top_k=50, seed=0):
    """Continue a prompt by n tokens. temp=0 means greedy (always the single most likely token)."""
    m = _m()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.tensor([enc.encode(prompt)], device=dev)
    for _ in range(n):
        logits = m(x[:, -m.config.block_size:])[0][0, -1, :V].float()
        if temp == 0:
            nxt = logits.argmax().view(1, 1)
        else:
            probs = F.softmax(logits / temp, -1).cpu()
            tp, ti = probs.topk(top_k)
            nxt = ti[torch.multinomial(tp, 1, generator=g)].view(1, 1).to(dev)
        x = torch.cat([x, nxt], 1)
    return enc.decode(x[0].tolist())


def sampling_lab(prompt, temps=(0, 0.4, 0.8, 1.2, 1.8), n=50, seed=0):
    """Same prompt, same random seed, different temperatures."""
    for t in temps:
        print(f"--- temperature {t} ---\n{complete(prompt, n=n, temp=t, seed=seed)}\n")


# ---------------------------------------------------------------------------
# 4. multiple-choice by likelihood (the same mechanism HellaSwag and ARC use)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score(prompt, options, m=None):
    """Rank continuations by how likely the model finds them. Start each option with a space, e.g. ' Paris'.
    'avg' is the per-token average log-probability (higher = more likely), 'total' is the sum."""
    m = _m(m)
    p = enc.encode(prompt)
    res = []
    for o in options:
        x = torch.tensor([p + enc.encode(o)], device=dev)
        lp = F.log_softmax(m(x)[0][0, :-1, :V].float(), -1)
        tok_lp = lp.gather(1, x[0, 1:, None])[:, 0]
        cont = tok_lp[len(p) - 1:]                     # only the tokens of the option
        res.append((o, cont.sum().item(), cont.mean().item()))
    res.sort(key=lambda r: -r[2])
    print(f"prompt: {prompt!r}")
    for o, s, a in res:
        print(f"  avg {a:7.3f} | total {s:8.2f} | {o!r}")
    return res


# ---------------------------------------------------------------------------
# 5. can it copy? (induction heads)
# ---------------------------------------------------------------------------
@torch.no_grad()
def copy_test(m=None, n=64, L=50, seed=0):
    """Feed random tokens followed by the SAME random tokens again.
    The first copy is unpredictable (loss ~ ln(vocab) ~ 10.8 or worse). If the model has learned to copy
    from earlier in the context ('induction'), the loss on the second copy collapses."""
    m = _m(m)
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 50000, (n, L), generator=g)
    x = torch.cat([r, r], 1).to(dev)
    logits = m(x)[0][:, :-1, :V].float()
    ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
    return ce[:, :L - 1].mean().item(), ce[:, L:].mean().item()


def copy_over_training(pattern="log/model_[0-9]*.pt"):
    """Run copy_test on every snapshot (every 5000 steps) and plot when the copying ability appears."""
    rows = []
    for p in sorted(glob.glob(pattern)):
        m, meta = load_ours(p)
        m.to(dev).eval()
        a, b = copy_test(m)
        rows.append((meta.get("step"), a, b))
        print(f"step {meta.get('step'):>6}  first copy {a:6.2f}   repeated copy {b:6.2f}")
        del m
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="first copy (random)")
        plt.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="repeated copy")
        plt.xlabel("training step"); plt.ylabel("loss (nats)"); plt.grid(alpha=.3); plt.legend()
        plt.title("When does the model learn to copy?")
        plt.show()
    except ImportError:
        pass
    return rows


# ---------------------------------------------------------------------------
# 6. which layers matter? (ablation)
# ---------------------------------------------------------------------------
def layer_ablation(max_tokens=200_000, data_root="edu_fineweb10B"):
    """Skip one transformer block at a time and measure how much the validation loss gets worse."""
    m = _m()
    val, tb = load_val_tokens(data_root), token_bytes_table(enc)

    def loss():
        with contextlib.redirect_stdout(io.StringIO()):
            return window_loss(m, val, dev, 8, 1024, max_tokens, tb)["loss"]

    base = loss()
    print(f"baseline val loss: {base:.4f}")
    deltas = []
    for i, blk in enumerate(m.transformer.h):
        h = blk.register_forward_hook(lambda mod, inp, out: inp[0])   # returning inp[0] = skip this block
        d = loss() - base
        h.remove()
        deltas.append(d)
        print(f"  skip layer {i:2d}: loss +{d:.3f}")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3.5))
        plt.bar(range(len(deltas)), deltas)
        plt.xlabel("layer skipped"); plt.ylabel("increase in val loss"); plt.grid(axis="y", alpha=.3)
        plt.title("How much does each layer matter?")
        plt.show()
    except ImportError:
        pass
    return deltas