import os
import math
import time
import inspect
import argparse
from dataclasses import dataclass, asdict
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------
# simple launch:
#   python train_gpt2.py                 # fresh run
#   python train_gpt2.py --resume        # continue from log/ckpt_latest.pt
#   python train_gpt2.py --no_compile    # debug without torch.compile
# DDP launch for e.g. 8 GPUs:
#   torchrun --standalone --nproc_per_node=8 train_gpt2.py
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # flash kernel on Ampere+, memory-efficient kernel on T4 (picked automatically)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use a "Conv1D" module, we use a vanilla Linear -> transpose these weights
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)  # int32 in memory (half of int64); cast to long per batch
    return torch.from_numpy(npt)

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, verbose=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if verbose:
            print(f"found {len(shards)} shards for split {split}")
        self.tokens = None
        self.loaded_shard = -1
        self.reset()

    def _load(self, shard_idx):
        # only hit the disk if the requested shard is not already in memory
        if self.loaded_shard != shard_idx:
            self.tokens = load_tokens(self.shards[shard_idx])
            self.loaded_shard = shard_idx

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self._load(0)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1].long()
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._load(self.current_shard)
            self.current_position = B * T * self.process_rank
        return x, y

    # checkpoint support: store a rank-independent position so any rank can restore it
    def state_dict(self):
        return {'shard': self.current_shard,
                'base_pos': self.current_position - self.B * self.T * self.process_rank}

    def load_state_dict(self, sd):
        self.current_shard = sd['shard']
        self._load(self.current_shard)
        self.current_position = sd['base_pos'] + self.B * self.T * self.process_rank

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    logits = logits.float()  # compute the loss in fp32, fp16 logits are too coarse for close calls
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # the completion with the lowest average loss is the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="resume from log/ckpt_latest.pt if it exists")
parser.add_argument("--no_compile", action="store_true", help="disable torch.compile")
args = parser.parse_args()

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "DDP needs CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
# T4 (Turing) has no native bf16 and no TF32 -> fp16 autocast + GradScaler everywhere
amp_dtype = torch.float16

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# ---- run configuration ------------------------------------------------------
total_batch_size = 32768          # tokens per optimizer step (2^15)
B = 16                          # micro batch size
T = 1024                          # sequence length
target_tokens = 2_500_000_000       # total training budget in tokens -> decides max_steps
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_frac = 0.0375              # fraction of steps used for LR warmup
weight_decay = 0.1
val_every = 250                   # validation loss
val_loss_steps = 20
hella_every = 2000                # HellaSwag (slow: ~10k examples)
sample_every = 500                # text generation
ckpt_every = 1000                 # rolling full checkpoint (must be a multiple of val_every)
snapshot_every = 5000             # extra weights-only snapshots that are never overwritten
use_compile = not args.no_compile
# -----------------------------------------------------------------------------
assert ckpt_every % val_every == 0, "checkpoints are written inside the validation block"
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
max_steps = target_tokens // total_batch_size
warmup_steps = int(warmup_frac * max_steps)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"=> max_steps: {max_steps} ({max_steps * total_batch_size / 1e6:.0f}M tokens), warmup_steps: {warmup_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", verbose=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", verbose=master_process)

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
orig_model = model                      # uncompiled module; shares the SAME parameters as the compiled one
if use_compile:
    model = torch.compile(model)        # used for training + fixed-shape validation
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = orig_model                  # clean state_dict keys (no _orig_mod. / module. prefixes)
# HellaSwag + generation use orig_model: variable sequence shapes would make torch.compile recompile constantly

def get_lr(it):
    # 1. linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2. after the schedule ends, stay at min lr
    if it > max_steps:
        return min_lr
    # 3. in between, cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1, goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, verbose=master_process)
scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))

# log dir + checkpointing
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
latest_path = os.path.join(log_dir, "ckpt_latest.pt")

def save_checkpoint(path, step, val_loss=None, full=True):
    ckpt = {
        'model': raw_model.state_dict(),
        'config': asdict(raw_model.config),
        'step': step,
        'val_loss': val_loss,
    }
    if full:
        ckpt['optimizer'] = optimizer.state_dict()
        ckpt['scaler'] = scaler.state_dict()
        ckpt['train_loader'] = train_loader.state_dict()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save can never corrupt the previous checkpoint

# resume
start_step = 0
resumed = False
if args.resume and os.path.exists(latest_path):
    ckpt = torch.load(latest_path, map_location=device, weights_only=True)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    train_loader.load_state_dict(ckpt['train_loader'])
    start_step = ckpt['step']  # checkpoint was written before this step ran, so redo it
    resumed = True
    del ckpt
    if master_process:
        print(f"resumed from {latest_path} at step {start_step}")
elif args.resume and master_process:
    print(f"--resume given but {latest_path} not found, starting fresh")

if master_process:
    with open(log_file, "a" if resumed else "w") as f:
        pass

for step in range(start_step, max_steps):
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss (compiled model, fixed shapes)
    if step % val_every == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.zeros((), device=device)
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach() / val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and step != start_step and step % ckpt_every == 0:
                save_checkpoint(latest_path, step, val_loss_accum.item(), full=True)
                if step % snapshot_every == 0:
                    save_checkpoint(os.path.join(log_dir, f"model_{step:05d}.pt"), step, val_loss_accum.item(), full=False)

    # once in a while evaluate hellaswag (skip step 0: random init, wasted minutes)
    if (step > 0 and step % hella_every == 0) or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = orig_model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % sample_every == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = orig_model(xgen) # (B, T, vocab_size)
                logits = logits[:, -1, :].float() # (B, vocab_size), softmax in fp32
                probs = F.softmax(logits, dim=-1)
                # top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # ---- one optimization step ----
    model.train()
    t0 = time.time()  # start the clock AFTER eval so tok/sec is not polluted by eval time
    loss_accum = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = model(x, y)
        # scale the loss to account for gradient accumulation (we want a MEAN, gradients just add up)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    scaler.step(optimizer)           # silently skipped by GradScaler if grads contain inf/nan
    scale_before = scaler.get_scale()
    scaler.update()
    skipped = scaler.get_scale() < scale_before  # scale only drops when a step was skipped

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
              f" | scale: {scaler.get_scale():.0f}{' | SKIPPED (inf/nan grads)' if skipped else ''}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# final checkpoint AFTER the last optimizer step (the in-loop ones are written before their step runs)
if master_process:
    save_checkpoint(latest_path, max_steps, None, full=True)
    save_checkpoint(os.path.join(log_dir, "model_final.pt"), max_steps, None, full=False)
    print("training finished, saved log/model_final.pt")

if ddp:
    destroy_process_group()