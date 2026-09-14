"""Reference recipe for the RoboTwin subtasks: frozen DINOv2 features from the head
and front cameras, a transformer head trained with Muon on 14-dim joint chunks,
word-token language over the demonstrations' paraphrases, served over the
policy socket.

    NANOVLA_MODE=train NANOVLA_OUTPUT_DIR=out python solution.py
    NANOVLA_MODE=serve NANOVLA_OUTPUT_DIR=out NANOVLA_SOCKET=/run/nanovla-eval/p.sock python solution.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUBTASK = "03-robotwin-open-design"
SHARDS = "/assets/nanovla/shards"
RECIPES = {
    "03-robotwin-open-design": dict(budget=1800, dim=384, depth=8, heads=6, chunk=16, steps=14000, batch=512, noise=0.0),
    "04-robotwin-robustness": dict(budget=1800, dim=384, depth=8, heads=6, chunk=16, steps=14000, batch=512, noise=0.15),
}
POOL, EMB_DIM, TEXT_LEN, SEED = 4, 768, 32, 0
TOWER = "dinov2"
STATE_DIM, ACT_DIM = 16, 14


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
    """[text tokens | head feature tokens | front feature tokens | state | action queries]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["dim"]
        self.text_embed = nn.Embedding(cfg["vocab_size"], d)
        self.patch_embed = nn.Linear(EMB_DIM, d)
        self.cam_embed = nn.Parameter(torch.zeros(2, 1, d))
        self.state_proj = nn.Linear(STATE_DIM, d)
        self.act_queries = nn.Parameter(torch.zeros(cfg["chunk"], d))
        self.pos = nn.Parameter(torch.zeros(TEXT_LEN + 2 * POOL * POOL + 1 + cfg["chunk"], d))
        self.blocks = nn.ModuleList(Block(d, cfg["heads"]) for _ in range(cfg["depth"]))
        self.norm_f = nn.RMSNorm(d)
        self.head = nn.Linear(d, ACT_DIM)
        self.apply(self._init)
        for p in (self.act_queries, self.pos, self.cam_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.proj.weight)
            nn.init.zeros_(blk.fc2.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, text_ids, feats, state):
        B = text_ids.size(0)
        toks = [self.text_embed(text_ids)]
        for c in range(2):
            toks.append(self.patch_embed(feats[:, c]) + self.cam_embed[c])
        toks.append(self.state_proj(state).unsqueeze(1))
        toks.append(self.act_queries.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(toks, 1) + self.pos
        for blk in self.blocks:
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


def words(text):
    return "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()


def tokenize(text, vocab):
    ids = [vocab.get(w, 1) for w in words(text)][:TEXT_LEN]
    return ids + [0] * (TEXT_LEN - len(ids))


def normalize(x, st):
    lo, hi = np.array(st["q01"], np.float32), np.array(st["q99"], np.float32)
    return np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -3.0, 3.0).astype(np.float32)


def extract_features(device, log):
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


def train():
    recipe = RECIPES[SUBTASK]
    out_dir = os.environ["NANOVLA_OUTPUT_DIR"]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.load(open(f"{SHARDS}/meta.json"))
    N = meta["n_frames"]
    stats = json.load(open(f"{SHARDS}/norm_stats.json"))
    instructions = json.load(open(f"{SHARDS}/instructions.json"))  # per episode: paraphrase list
    vocab = {w: i + 2 for i, w in enumerate(sorted({w for lst in instructions for s in lst for w in words(s)}))}
    per_episode = max(len(lst) for lst in instructions)
    tokens = torch.zeros(len(instructions), per_episode, TEXT_LEN, dtype=torch.long)
    counts = torch.zeros(len(instructions), dtype=torch.long)
    for e, lst in enumerate(instructions):
        counts[e] = len(lst)
        for i, s in enumerate(lst):
            tokens[e, i] = torch.tensor(tokenize(s, vocab))
    state = torch.from_numpy(normalize(np.fromfile(f"{SHARDS}/state.f32", np.float32).reshape(N, STATE_DIM), stats["state"]))
    actions = torch.from_numpy(normalize(np.fromfile(f"{SHARDS}/actions.f32", np.float32).reshape(N, ACT_DIM), stats["actions"]))
    ep_end = torch.from_numpy(np.fromfile(f"{SHARDS}/ep_end.i32", np.int32).astype(np.int64))
    episode = torch.from_numpy(np.fromfile(f"{SHARDS}/episode_id.i32", np.int32).astype(np.int64))
    feats = extract_features(device, log).to(device)
    state, actions, ep_end, episode, tokens, counts = (x.to(device) for x in (state, actions, ep_end, episode, tokens, counts))

    cfg = {"dim": recipe["dim"], "depth": recipe["depth"], "heads": recipe["heads"],
           "chunk": recipe["chunk"], "vocab_size": len(vocab) + 2}
    model = Policy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"policy {n_params / 1e6:.2f}M params, vocab {len(vocab)}, {N} frames")
    matrix, other = model.param_groups()
    opts = [Muon(matrix, lr=0.02), torch.optim.AdamW(other, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED * 7777)
    chunk, bs, steps = recipe["chunk"], recipe["batch"], recipe["steps"]
    offs = torch.arange(chunk, device=device)
    deadline = t_start + recipe["budget"] - 90
    ema, step = None, 0
    while True:
        progress = max(step / steps, (time.time() - t_start) / (deadline - t_start))
        if progress >= 1.0:
            break
        idx = torch.randint(0, N, (bs,), device=device, generator=gen)
        raw = idx[:, None] + offs
        end = ep_end[idx][:, None]
        tgt = torch.minimum(raw, end - 1)
        mask = (raw < end).float()
        ep = episode[idx]
        pick = (torch.rand(bs, device=device, generator=gen) * counts[ep]).long()  # random paraphrase
        text = tokens[ep, pick]
        x = feats[idx].float()
        if recipe["noise"] > 0:
            x = x + recipe["noise"] * torch.randn(x.shape, device=device, generator=gen)
        mult = min(1.0, (step + 1) / 100) * (1.0 if progress < 0.6 else 1.0 - 0.95 * (progress - 0.6) / 0.4)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * mult
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(text, x, state[idx])
            loss = ((pred - actions[tgt]).abs() * mask[..., None]).sum() / (mask.sum() * ACT_DIM)
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
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "ckpt.pt"))
    json.dump({"cfg": cfg, "vocab": vocab, "stats": stats, "tower": TOWER, "pool": POOL, "steps": step,
               "params": n_params}, open(os.path.join(out_dir, "meta.json"), "w"))
    log("saved ckpt.pt + meta.json")


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
    cache = {}

    def infer(requests):
        B = len(requests)
        for r in requests:
            if r.language not in cache:
                cache[r.language] = torch.tensor(tokenize(r.language, vocab))
        text = torch.stack([cache[r.language] for r in requests]).to(device)
        cams = np.concatenate([r.cameras for r in requests])
        with torch.no_grad():
            feats = encode(cams).view(B, 2, POOL * POOL, EMB_DIM)
            st = torch.from_numpy(normalize(np.stack([r.state for r in requests]), stats["state"])).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                chunk = model(text, feats, st).float().cpu().numpy()
        chunk = np.clip(chunk, -1.0, 1.0)  # joint targets: stay inside the demonstrated range
        return (chunk + 1) / 2 * np.maximum(a_hi - a_lo, 1e-6) + a_lo

    serve_socket(os.environ["NANOVLA_SOCKET"], infer, batch_size=16, batch_wait_ms=3.0)


if __name__ == "__main__":
    serve() if os.environ.get("NANOVLA_MODE", "train") == "serve" else train()
