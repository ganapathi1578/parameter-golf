# ============================================================
# PARAMETER GOLF — CROSS-HATCH V2 STABLE
# 2048 = 128 × 16
#
# Factorized architecture submission for Parameter Golf.
# Non-record 16MB track.
#
# Main features:
#   1. tied embedding init std = 0.005
#   2. logit softcap = 30
#   3. no weight decay
#   4. Muon + Adam split
#   5. T4 -> FP16 + GradScaler (Ampere+ -> BF16)
#   6. official eval_val BPB
#   7. DDP distributed training
#   8. pure FP16 serialization (fits in 16MB)
# ============================================================

import os
import sys
import math
import time
import glob
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

# ============================================================
# DISTRIBUTED SETUP
# ============================================================

dist_initialized = False
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    dist.init_process_group(backend="nccl")
    dist_initialized = True
    torch.cuda.set_device(LOCAL_RANK)

def log0(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs)

# ============================================================
# CONFIG
# ============================================================

TRAIN_SHARDS = int(os.environ.get("TRAIN_SHARDS", "20"))
SEQ_LEN = 1024
VOCAB_SIZE = 1024

NUM_LAYERS = 9
CHANNEL_DIM = 128
NUM_LANES = 16
MODEL_DIM = 2048
FFN_DIM = 1408

MAX_WALLCLOCK_SEC = int(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))
WARMUP_STEPS = 20
WARMDOWN_START = 0.72

# H100 Target Global Batch
GLOBAL_BATCH_TOKENS = 524288
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "32"))

# Ensure micro_batch is valid for the global batch
_tokens_per_micro = WORLD_SIZE * MICRO_BATCH_SIZE * SEQ_LEN
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", str(max(1, GLOBAL_BATCH_TOKENS // _tokens_per_micro))))

EFFECTIVE_BATCH = WORLD_SIZE * MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
if EFFECTIVE_BATCH != GLOBAL_BATCH_TOKENS:
    log0(f"WARNING: Effective batch {EFFECTIVE_BATCH} != target {GLOBAL_BATCH_TOKENS}")

MUON_LR = 0.025
TIED_EMBED_LR = 0.05
SCALAR_LR = 0.04

MUON_MOMENTUM = 0.99
MUON_MOMENTUM_START = 0.92
MUON_MOMENTUM_WARMUP = 1500

BETAS = (0.9, 0.95)
EPS = 1e-8
GRAD_CLIP = 0.0

EMA_DECAY = 0.9965
EMA_START_FRAC = 0.72

QK_GAIN_INIT = 1.5
ROPE_BASE = 10000.0
LOGIT_SOFTCAP = 30.0

ATTN_VECTOR_MIX = True
FFN_PRE_MIX = False
FFN_POST_MIX = True
PER_LANE_SCALE = True

USE_COMPILE = True
COMPILE_MODE = "max-autotune"

SEED = int(os.environ.get("SEED", "1337"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _ in range(10):
    if (_REPO_ROOT / "data" / "datasets").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

DATA_DIR = Path(os.environ.get("DATA_PATH", str(_REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", str(_REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")))

# ============================================================
# GPU & AMP
# ============================================================

log0("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

device = torch.device(f"cuda:{LOCAL_RANK}")
gpu_name = torch.cuda.get_device_name(LOCAL_RANK)
gpu_cc = torch.cuda.get_device_capability(LOCAL_RANK)

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

USE_BF16 = gpu_cc[0] >= 8
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_GRAD_SCALER = not USE_BF16

if USE_BF16:
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

log0("AMP dtype:", AMP_DTYPE)
log0("GradScaler:", USE_GRAD_SCALER)
log0("World size:", WORLD_SIZE)
log0("Micro batch:", MICRO_BATCH_SIZE)
log0("Grad accumulation:", GRAD_ACCUM)
log0("Global batch tokens:", EFFECTIVE_BATCH)

def amp_context():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)

# ============================================================
# DATA LOADING
# ============================================================

train_files = sorted(glob.glob(str(DATA_DIR / "fineweb_train_*.bin")))[:TRAIN_SHARDS]
val_files = sorted(glob.glob(str(DATA_DIR / "fineweb_val_*.bin")))

if not train_files or not val_files:
    raise RuntimeError("Dataset shards not found.")

log0("Train shards:", len(train_files))
log0("Val shards:", len(val_files))

HEADER_BYTES = 256 * 4
MAGIC = 20240520
VERSION = 1

class MemmapShard:
    def __init__(self, path):
        self.path = Path(path)
        header = np.fromfile(self.path, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != MAGIC or int(header[1]) != VERSION:
            raise RuntimeError(f"Bad shard: {self.path}")
        self.n = int(header[2])
        self.arr = np.memmap(self.path, mode="r", dtype="<u2", offset=HEADER_BYTES, shape=(self.n,))

    def read(self, start, count):
        end = min(start + count, self.n)
        return np.array(self.arr[start:end], dtype=np.uint16, copy=True)

class DistributedStreamLoader:
    def __init__(self, files, rank, world_size):
        self.shards = [MemmapShard(f) for f in files]
        self.si = 0
        self.pos = 0
        self.rank = rank
        self.world_size = world_size

    def take_global(self, n):
        parts = []
        remaining = n
        while remaining > 0:
            shard = self.shards[self.si]
            available = shard.n - self.pos
            if available <= 0:
                self.si = (self.si + 1) % len(self.shards)
                self.pos = 0
                continue
            count = min(remaining, available)
            parts.append(shard.read(self.pos, count))
            self.pos += count
            remaining -= count
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def next_batch(self, micro_batch, seq_len):
        global_tokens = (micro_batch * self.world_size) * seq_len + 1
        raw = self.take_global(global_tokens)
        t = torch.from_numpy(raw.astype(np.int64, copy=False))
        
        start = self.rank * micro_batch * seq_len
        end = start + micro_batch * seq_len + 1
        local_t = t[start:end]
        
        x = local_t[:-1].reshape(micro_batch, seq_len)
        y = local_t[1:].reshape(micro_batch, seq_len)
        return x.pin_memory(), y.pin_memory()

train_loader = DistributedStreamLoader(train_files, RANK, WORLD_SIZE)

# ============================================================
# TOKENIZER & OFFICIAL BPB LUTS
# ============================================================

sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
if sp.vocab_size() != VOCAB_SIZE:
    raise RuntimeError("Tokenizer vocab mismatch.")

def build_sentencepiece_luts(sp_model, vocab_size, dev):
    sp_vocab_size = int(sp_model.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp_model.is_control(token_id) or sp_model.is_unknown(token_id) or sp_model.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp_model.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp_model.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=dev),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=dev),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=dev),
    )

base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE, device)

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([torch.from_numpy(MemmapShard(f).read(0, MemmapShard(f).n).astype(np.int64)) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

val_tokens = load_validation_tokens(str(DATA_DIR / "fineweb_val_*.bin"), SEQ_LEN)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype), self.eps)

class SharedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, zero_init=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim, dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.orthogonal_(self.weight)
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        y = x @ self.weight.to(dtype=x.dtype)
        return y.reshape(*shape[:-1], self.weight.size(1))

class RoPE(nn.Module):
    def __init__(self, dim, seq_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freq = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cache", freq.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", freq.sin()[None, None, :, :], persistent=False)
    def forward(self, x):
        T = x.size(-2)
        cos = self.cos_cache[:, :, :T, :].to(x.device, x.dtype)
        sin = self.sin_cache[:, :, :T, :].to(x.device, x.dtype)
        xe, xo = x[..., 0::2], x[..., 1::2]
        ye = xe * cos - xo * sin
        yo = xe * sin + xo * cos
        return torch.stack([ye, yo], dim=-1).flatten(-2)

class FactorizedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        C, V = CHANNEL_DIM, NUM_LANES
        self.q = SharedLinear(C, C)
        self.k = SharedLinear(C, C)
        self.v = SharedLinear(C, C)
        self.o = SharedLinear(C, C, zero_init=True)
        self.q_norm = RMSNorm(C)
        self.k_norm = RMSNorm(C)
        self.q_gain = nn.Parameter(torch.full((V,), QK_GAIN_INIT, dtype=torch.float32))
        self.vector_mix = nn.Parameter(torch.eye(V, dtype=torch.float32) + 0.01 * torch.randn(V, V)) if ATTN_VECTOR_MIX else None
        self.rope = RoPE(C, SEQ_LEN, ROPE_BASE)

    def forward(self, x):
        B, T, C, V = x.shape
        xf = x.reshape(B * T * V, C)
        q = (xf @ self.q.weight.to(x.dtype)).reshape(B, T, V, C)
        k = (xf @ self.k.weight.to(x.dtype)).reshape(B, T, V, C)
        v = (xf @ self.v.weight.to(x.dtype)).reshape(B, T, V, C)
        q, k = self.q_norm(q).permute(0, 2, 1, 3), self.k_norm(k).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.permute(0, 2, 3, 1).contiguous()
        if self.vector_mix is not None:
            y = torch.matmul(y, self.vector_mix.to(y.dtype))
        y = y.permute(0, 1, 3, 2).contiguous()
        out = (y.reshape(B * T * V, C) @ self.o.weight.to(y.dtype)).reshape(B, T, V, C)
        return out.permute(0, 1, 3, 2).contiguous()

class CrossHatchSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        C, V, FFD = CHANNEL_DIM, NUM_LANES, FFN_DIM
        self.gate = SharedLinear(C, FFD)
        self.up = SharedLinear(C, FFD)
        self.down = SharedLinear(FFD, C, zero_init=True)
        self.pre_mix = nn.Parameter(torch.eye(V, dtype=torch.float32) + 0.01 * torch.randn(V, V)) if FFN_PRE_MIX else None
        self.post_mix = nn.Parameter(torch.eye(V, dtype=torch.float32) + 0.01 * torch.randn(V, V)) if FFN_POST_MIX else None

    def forward(self, x):
        B, T, C, V = x.shape
        if self.pre_mix is not None:
            x = torch.matmul(x, self.pre_mix.to(x.dtype))
        xf = x.reshape(B * T * V, C)
        gate, up = xf @ self.gate.weight.to(x.dtype), xf @ self.up.weight.to(x.dtype)
        hidden = F.silu(gate) * up
        out = (hidden @ self.down.weight.to(x.dtype)).reshape(B, T, V, C).permute(0, 1, 3, 2).contiguous()
        if self.post_mix is not None:
            out = torch.matmul(out, self.post_mix.to(out.dtype))
        return out

class CrossHatchBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(MODEL_DIM), RMSNorm(MODEL_DIM)
        self.attn, self.ffn = FactorizedAttention(), CrossHatchSwiGLU()
        self.attn_alpha, self.ffn_alpha = nn.Parameter(torch.ones(1, dtype=torch.float32)), nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.lane_scale = nn.Parameter(torch.ones(NUM_LANES, dtype=torch.float32)) if PER_LANE_SCALE else None

    def forward(self, x):
        B, T, C, V = x.shape
        z = self.attn_norm(x.reshape(B, T, MODEL_DIM)).reshape_as(x)
        if self.lane_scale is not None: z = z * self.lane_scale.to(z.dtype)[None, None, None, :]
        x = x + self.attn_alpha.to(z.dtype) * self.attn(z)
        z = self.ffn_norm(x.reshape(B, T, MODEL_DIM)).reshape_as(x)
        if self.lane_scale is not None: z = z * self.lane_scale.to(z.dtype)[None, None, None, :]
        x = x + self.ffn_alpha.to(z.dtype) * self.ffn(z)
        return x

class CrossHatchGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.input_norm = RMSNorm(MODEL_DIM)
        self.blocks = nn.ModuleList([CrossHatchBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(MODEL_DIM)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        for module in self.modules():
            if isinstance(module, SharedLinear) and not torch.all(module.weight == 0):
                nn.init.orthogonal_(module.weight)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        for block in self.blocks:
            nn.init.zeros_(block.attn.o.weight)
            nn.init.zeros_(block.ffn.down.weight)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.input_norm(self.tok_emb(input_ids).to(AMP_DTYPE)).reshape(B, T, CHANNEL_DIM, NUM_LANES)
        for block in self.blocks: x = block(x)
        h = self.final_norm(x.reshape(B, T, MODEL_DIM))
        logits = LOGIT_SOFTCAP * torch.tanh(F.linear(h.float(), self.tok_emb.weight.float()) / LOGIT_SOFTCAP)
        if targets is None: return logits
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

model = CrossHatchGPT().to(device)

if dist_initialized:
    model = DDP(model, device_ids=[LOCAL_RANK], broadcast_buffers=False)
base_model = model.module if dist_initialized else model

# ============================================================
# OPTIMIZER (MUON + ADAM)
# ============================================================

def zeropower_ns5(G, steps=10, eps=1e-7):
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed: X = X.T
    X = X / (X.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed: X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, momentum_start, warmup_steps):
        defaults = dict(lr=lr, momentum=momentum, momentum_start=momentum_start, warmup_steps=warmup_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            step_num, warmup = group["step"], group["warmup_steps"]
            momentum = group["momentum_start"] + min(1.0, step_num / warmup) * (group["momentum"] - group["momentum_start"]) if warmup > 0 else group["momentum"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.detach().float()
                state = self.state[p]
                if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)
                update = zeropower_ns5(buf) * (max(1.0, p.shape[0]/p.shape[1]) ** 0.5)
                p.add_(update.to(p.dtype), alpha=-group["lr"])

muon_params, embed_params, scalar_params = [], [base_model.tok_emb.weight], []
muon_ids = set()

for name, p in base_model.named_parameters():
    if name == "tok_emb.weight": continue
    if p.ndim == 2 and p.numel() > 4096:
        muon_params.append(p)
        muon_ids.add(id(p))

scalar_params = [p for name, p in base_model.named_parameters() if name != "tok_emb.weight" and id(p) not in muon_ids]

muon = Muon(muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM, momentum_start=MUON_MOMENTUM_START, warmup_steps=MUON_MOMENTUM_WARMUP)
adam_embed = torch.optim.Adam([{"params": embed_params, "lr": TIED_EMBED_LR}], betas=BETAS, eps=EPS, fused=True)
adam_scalar = torch.optim.Adam([{"params": scalar_params, "lr": SCALAR_LR}], betas=BETAS, eps=EPS, fused=True)

try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
except AttributeError:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER)

if USE_COMPILE and gpu_cc[0] >= 8:
    try:
        model = torch.compile(model, mode=COMPILE_MODE)
        log0("torch.compile: ENABLED")
    except Exception:
        pass

def lr_mult(elapsed, step):
    if step < WARMUP_STEPS: return max(1e-3, step / max(1, WARMUP_STEPS))
    frac = elapsed / MAX_WALLCLOCK_SEC
    if frac <= WARMDOWN_START: return 1.0
    t = min(1.0, max(0.0, (frac - WARMDOWN_START) / (1.0 - WARMDOWN_START)))
    return 0.5 * (1.0 + math.cos(math.pi * t))

def set_lrs(elapsed, step):
    m = lr_mult(elapsed, step)
    for g in muon.param_groups: g["lr"] = MUON_LR * m
    for g in adam_embed.param_groups: g["lr"] = TIED_EMBED_LR * m
    for g in adam_scalar.param_groups: g["lr"] = SCALAR_LR * m
    return m

ema_shadow = {}
ema_initialized = False

def update_ema():
    global ema_initialized
    with torch.no_grad():
        if not ema_initialized:
            for name, p in base_model.named_parameters():
                ema_shadow[name] = p.detach().float().clone()
            ema_initialized = True
        else:
            for name, p in base_model.named_parameters():
                ema_shadow[name].lerp_(p.detach().float(), 1.0 - EMA_DECAY)

# ============================================================
# OFFICIAL VALIDATION
# ============================================================

@torch.no_grad()
def eval_val():
    model.eval()
    local_batch_seqs = max(1, MICRO_BATCH_SIZE)
    total_seqs = (val_tokens.numel() - 1) // SEQ_LEN
    seq_start = (total_seqs * RANK) // WORLD_SIZE
    seq_end = (total_seqs * (RANK + 1)) // WORLD_SIZE
    
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * SEQ_LEN
            raw_end = batch_end * SEQ_LEN + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, SEQ_LEN), local[1:].reshape(-1, SEQ_LEN)
            with amp_context():
                batch_loss = model(x, y).detach()
            batch_tokens = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_tokens
            val_token_count += batch_tokens
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist_initialized:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# ============================================================
# TRAINING
# ============================================================

log0("\n=== TRAINING ===")
torch.cuda.synchronize()
start = time.perf_counter()
optimizer_step = 0
tokens_seen = 0

model.train()
while True:
    elapsed = time.perf_counter() - start
    if elapsed >= MAX_WALLCLOCK_SEC: break
    
    muon.zero_grad(set_to_none=True)
    adam_embed.zero_grad(set_to_none=True)
    adam_scalar.zero_grad(set_to_none=True)
    set_lrs(elapsed, optimizer_step)

    step_loss, completed_micro = 0.0, 0
    
    for _ in range(GRAD_ACCUM):
        if time.perf_counter() - start >= MAX_WALLCLOCK_SEC: break
        x, y = train_loader.next_batch(MICRO_BATCH_SIZE, SEQ_LEN)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with amp_context():
            loss = model(x, y)
        scaled_loss = loss / GRAD_ACCUM
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        step_loss += loss.detach().float().item()
        tokens_seen += (y.numel() * WORLD_SIZE)
        completed_micro += 1

    if completed_micro != GRAD_ACCUM: break

    if scaler.is_enabled():
        scaler.unscale_(muon)
        scaler.unscale_(adam_embed)
        scaler.unscale_(adam_scalar)
        
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP if GRAD_CLIP > 0 else float("inf"))
    
    if scaler.is_enabled():
        scaler.step(muon)
        scaler.step(adam_embed)
        scaler.step(adam_scalar)
        scaler.update()
    else:
        muon.step()
        adam_embed.step()
        adam_scalar.step()

    optimizer_step += 1
    elapsed = time.perf_counter() - start
    if elapsed / MAX_WALLCLOCK_SEC >= EMA_START_FRAC: update_ema()
    
    if RANK == 0 and optimizer_step % 10 == 0:
        log0(f"step={optimizer_step:4d} time={elapsed:6.2f}s loss={step_loss / GRAD_ACCUM:.5f} tok/s={tokens_seen/elapsed:,.0f}")

torch.cuda.synchronize()
log0(f"Training finished. Time: {time.perf_counter() - start:.2f}s, Steps: {optimizer_step}")

# ============================================================
# EVALUATION & ARTIFACT (FP16 ONLY)
# ============================================================

log0("\n=== OFFICIAL EVALUATION ===")
raw_vloss, raw_bpb = eval_val()
log0(f"pre_quant_val_loss: {raw_vloss:.8f}")
log0(f"pre_quant_val_bpb: {raw_bpb:.8f}")

log0("\n=== FP16 ARTIFACT EXPORT ===")
if RANK == 0:
    code = Path(__file__).read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))
    
    # Simple FP16 serialization as requested
    fp16_state = {k: v.detach().to("cpu", dtype=torch.float16) for k, v in base_model.state_dict().items()}
    artifact_path = _SCRIPT_DIR / "final_model.fp16.pt"
    torch.save(fp16_state, artifact_path)
    model_bytes = artifact_path.stat().st_size
    
    log0("--------------------------------")
    log0("ARTIFACT AUDIT")
    log0("--------------------------------")
    log0(f"serialized bytes: {model_bytes:,}")
    log0(f"code bytes: {code_bytes:,}")
    log0(f"TOTAL: {model_bytes + code_bytes:,}")
    log0("LIMIT: 16,000,000")
    log0(f"HEADROOM: {16000000 - (model_bytes + code_bytes):,}")
    log0("PASS/FAIL: " + ("PASS" if model_bytes + code_bytes <= 16000000 else "FAIL"))
    log0("--------------------------------")

if dist_initialized:
    dist.barrier()

log0("\n=== ROUND-TRIP TEST ===")
# Reconstruct model from artifact
artifact_path = _SCRIPT_DIR / "final_model.fp16.pt"
reloaded_state = torch.load(artifact_path, map_location="cpu", weights_only=False)
# Convert back to appropriate training dtype before evaluation to ensure exact behavior matching
reloaded_state = {k: v.to(device, dtype=torch.float32) for k, v in reloaded_state.items()}
base_model.load_state_dict(reloaded_state, strict=True)
torch.cuda.synchronize()

rt_vloss, rt_bpb = eval_val()
log0(f"roundtrip_val_loss: {rt_vloss:.8f}")
log0(f"roundtrip_val_bpb: {rt_bpb:.8f}")

if dist_initialized:
    dist.destroy_process_group()
