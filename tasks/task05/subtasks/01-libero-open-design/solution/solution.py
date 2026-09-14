"""Reference recipe for the two task05 LIBERO subtasks: frozen DINOv2 features, a small
transformer head trained with Muon, served over the policy socket.

    NANOVLA_MODE=train NANOVLA_OUTPUT_DIR=out python solution.py
    NANOVLA_MODE=serve NANOVLA_OUTPUT_DIR=out NANOVLA_SOCKET=/run/nanovla-eval/p.sock python solution.py

SUBTASK selects the recipe; everything else is shared.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUBTASK = "01-libero-open-design"
SHARDS = "/assets/nanovla/shards"
RECIPES = {
    #                budget  dim depth heads chunk state  steps batch  noise  lr_head
    "01-libero-open-design": dict(budget=1800, dim=384, depth=8, heads=6, chunk=16, state=True,
                         steps=16000, batch=1024, noise=0.0, tie=1),
    "02-libero-robustness": dict(budget=1800, dim=384, depth=8, heads=6, chunk=16, state=True,
                          steps=16000, batch=1024, noise=0.15, tie=1),
}
POOL, EMB_DIM, TEXT_LEN, SEED = 4, 768, 32, 0
TOWER = "dinov2"


# ------------------------------------------------------------------ model

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(dim // heads), nn.RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.relu(self.fc1(self.norm2(x))).square())


class Policy(nn.Module):
    """[text tokens | agent feature tokens | wrist feature tokens | state | action queries]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["dim"]
        self.text_embed = nn.Embedding(cfg["vocab_size"], d)
        self.patch_embed = nn.Linear(EMB_DIM, d)
        self.cam_embed = nn.Parameter(torch.zeros(2, 1, d))
        self.state_proj = nn.Linear(8, d)
        self.act_queries = nn.Parameter(torch.zeros(cfg["chunk"], d))
        self.pos = nn.Parameter(torch.zeros(self.seq_len, d))
        self.blocks = nn.ModuleList(Block(d, cfg["heads"]) for _ in range(cfg["depth"] // cfg["tie"]))
        self.norm_f = nn.RMSNorm(d)
        self.head = nn.Linear(d, 7)
        self.apply(self._init)
        for p in (self.act_queries, self.pos, self.cam_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.proj.weight)
            nn.init.zeros_(blk.fc2.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def seq_len(self):
        return TEXT_LEN + 2 * POOL * POOL + 1 + self.cfg["chunk"]

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, text_ids, feats, state):
        """text_ids (B, TEXT_LEN) long; feats (B, 2, 16, 768); state (B, 8) normalised."""
        B = text_ids.size(0)
        toks = [self.text_embed(text_ids)]
        for c in range(2):
            toks.append(self.patch_embed(feats[:, c]) + self.cam_embed[c])
        if not self.cfg["state"]:
            state = state * 0.0
        toks.append(self.state_proj(state).unsqueeze(1))
        toks.append(self.act_queries.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(toks, 1) + self.pos
        for blk in self.blocks:
            for _ in range(self.cfg["tie"]):
                x = blk(x)
        return self.head(self.norm_f(x[:, -self.cfg["chunk"]:]))

    def param_groups(self):
        matrix, other = [], []
        for n, p in self.named_parameters():
            (matrix if p.ndim >= 2 and "blocks" in n else other).append(p)
        return matrix, other


def newton_schulz(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp_(buf, group["momentum"])
                g2 = g.reshape(g.size(0), -1)
                u = newton_schulz(g2).view_as(p)
                p.add_(u, alpha=-group["lr"] * max(1.0, g2.size(0) / g2.size(1)) ** 0.5)


# ------------------------------------------------------------------- data

def tokenize(text, vocab):
    words = "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()
    ids = [vocab.get(w, 1) for w in words][:TEXT_LEN]
    return ids + [0] * (TEXT_LEN - len(ids))


def normalize(x, st):
    lo, hi = np.array(st["q01"], np.float32), np.array(st["q99"], np.float32)
    return np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -3.0, 3.0).astype(np.float32)


def extract_features(device, log):
    """Frozen DINOv2 pooled tokens for every frame of both cameras: (N, 2, 16, 768) fp16 on CPU."""
    import towers
    meta = json.load(open(f"{SHARDS}/meta.json"))
    N, H = meta["n_frames"], meta["img_size"]
    encode = towers.load_vision(TOWER, pool=POOL, device=device)
    out = torch.empty(N, 2, POOL * POOL, EMB_DIM, dtype=torch.float16)
    for cam, name in enumerate(("agent.u8", "wrist.u8")):
        frames = np.memmap(f"{SHARDS}/{name}", np.uint8, "r", shape=(N, H, H, 3))
        t0 = time.time()
        for start in range(0, N, 512):
            out[start:start + 512, cam] = encode(np.asarray(frames[start:start + 512])).half().cpu()
        log(f"features {name}: {N} frames in {time.time() - t0:.0f}s")
    return out


# ------------------------------------------------------------------ train

def train_worker(rank, world, recipe, out_dir, port):
    ddp = world > 1
    if ddp:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_NVLS_ENABLE="0")
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    master = rank == 0
    t_start = time.time()

    def log(msg):
        if master:
            print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    torch.manual_seed(SEED * 1000 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.load(open(f"{SHARDS}/meta.json"))
    N = meta["n_frames"]
    stats = json.load(open(f"{SHARDS}/norm_stats.json"))
    task_strs = json.load(open(f"{SHARDS}/task_strs.json"))
    words = sorted({w for t in task_strs for w in
                    "".join(ch if ch.isalnum() else " " for ch in t.lower()).split()})
    vocab = {w: i + 2 for i, w in enumerate(words)}
    task_tokens = torch.tensor([tokenize(t, vocab) for t in task_strs])
    state = torch.from_numpy(normalize(np.fromfile(f"{SHARDS}/state.f32", np.float32).reshape(N, 8), stats["state"]))
    actions = torch.from_numpy(normalize(np.fromfile(f"{SHARDS}/actions.f32", np.float32).reshape(N, 7), stats["actions"]))
    ep_end = torch.from_numpy(np.fromfile(f"{SHARDS}/ep_end.i32", np.int32).astype(np.int64))
    task_idx = torch.from_numpy(np.fromfile(f"{SHARDS}/task_idx.u8", np.uint8).astype(np.int64))
    feats = extract_features(device, log)
    feats_gpu = feats.to(device)  # ~5 GB fp16
    state, actions, ep_end, task_idx = (x.to(device) for x in (state, actions, ep_end, task_idx))
    task_tokens = task_tokens.to(device)

    cfg = {"dim": recipe["dim"], "depth": recipe["depth"], "heads": recipe["heads"],
           "chunk": recipe["chunk"], "state": recipe["state"], "tie": recipe["tie"],
           "vocab_size": len(vocab) + 2}
    model = Policy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"policy {n_params / 1e6:.2f}M params, world {world}")
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if ddp else model
    matrix, other = model.param_groups()
    opts = [Muon(matrix, lr=0.02), torch.optim.AdamW(other, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED * 7777 + rank)
    chunk, bs = recipe["chunk"], recipe["batch"] // world
    offs = torch.arange(chunk, device=device)
    deadline = t_start + recipe["budget"] - 90  # leave time for the checkpoint write
    steps = recipe["steps"]
    ema = None
    step = 0
    while True:
        progress = max(step / steps, (time.time() - t_start) / (deadline - t_start))
        if progress >= 1.0:
            break
        idx = torch.randint(0, N, (bs,), device=device, generator=gen)
        raw = idx[:, None] + offs
        end = ep_end[idx][:, None]
        tgt = torch.minimum(raw, end - 1)
        mask = (raw < end).float()
        x = feats_gpu[idx].float()
        if recipe["noise"] > 0:
            x = x + recipe["noise"] * torch.randn(x.shape, device=device, generator=gen)
        mult = min(1.0, (step + 1) / 100) * (1.0 if progress < 0.6 else 1.0 - 0.95 * (progress - 0.6) / 0.4)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * mult
        with torch.autocast("cuda", torch.bfloat16):
            pred = train_model(task_tokens[task_idx[idx]], x, state[idx])
            loss = ((pred - actions[tgt]).abs() * mask[..., None]).sum() / (mask.sum() * 7)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        lv = loss.item()
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % 200 == 0:
            log(f"step {step}/{steps} loss {lv:.4f} ema {ema:.4f} lr x{mult:.2f}")
        step += 1
    log(f"trained {step} steps, ema loss {ema:.4f}")
    if master:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "ckpt.pt"))
        json.dump({"cfg": cfg, "vocab": vocab, "stats": stats, "tower": TOWER, "pool": POOL,
                   "steps": step, "params": n_params}, open(os.path.join(out_dir, "meta.json"), "w"))
        log("saved ckpt.pt + meta.json")
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def train():
    recipe = RECIPES[SUBTASK]
    out_dir = os.environ["NANOVLA_OUTPUT_DIR"]
    world = torch.cuda.device_count()
    if world > 1:
        torch.multiprocessing.spawn(train_worker, args=(world, recipe, out_dir, 29511), nprocs=world, join=True)
    else:
        train_worker(0, 1, recipe, out_dir, 29511)


# ------------------------------------------------------------------ serve

def serve():
    import towers
    from remote_protocol import serve as serve_socket
    out_dir = os.environ["NANOVLA_OUTPUT_DIR"]
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model = Policy(meta["cfg"]).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu", weights_only=True))
    model.eval()
    encode = towers.load_vision(meta["tower"], pool=meta["pool"], device=device)
    vocab, stats = meta["vocab"], meta["stats"]
    a_lo, a_hi = (np.array(stats["actions"][k], np.float32) for k in ("q01", "q99"))
    token_cache = {}

    def infer(requests):
        B = len(requests)
        for r in requests:
            if r.language not in token_cache:
                token_cache[r.language] = torch.tensor(tokenize(r.language, vocab))
        text = torch.stack([token_cache[r.language] for r in requests]).to(device)
        cams = np.concatenate([r.cameras for r in requests])  # (B*2, 128, 128, 3)
        with torch.no_grad():
            feats = encode(cams).view(B, 2, POOL * POOL, EMB_DIM)
            st = torch.from_numpy(normalize(np.stack([r.state for r in requests]), stats["state"])).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                chunk = model(text, feats, st).float().cpu().numpy()
        return np.clip((chunk + 1) / 2 * np.maximum(a_hi - a_lo, 1e-6) + a_lo, -1.0, 1.0)

    serve_socket(os.environ["NANOVLA_SOCKET"], infer, batch_size=32, batch_wait_ms=3.0)


if __name__ == "__main__":
    mode = os.environ.get("NANOVLA_MODE", "train")
    serve() if mode == "serve" else train()
