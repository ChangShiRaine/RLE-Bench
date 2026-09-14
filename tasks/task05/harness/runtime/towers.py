"""Loaders for the mounted encoder bundle, one API per tower kind.

Vision towers  -> encode(u8 (B,S,S,3)) -> (B, pool*pool, dim) float32
Text towers    -> encode([str, ...])   -> (T, 1, dim) float32

Only models present in the bundle under HF_HOME load; every tower is pinned to
one Hub revision. Fuse several vision towers by concatenating their outputs
along the last dimension.
"""

import re

import numpy as np
import torch
import torch.nn.functional as Fn

HF_REVISIONS = {
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "facebook/dinov2-giant": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
    "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "openai/clip-vit-base-patch16": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "google/flan-t5-small": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    "google/flan-t5-base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
}


def _tower(hf, **kwargs):
    return dict(hf=hf, revision=HF_REVISIONS[hf], **kwargs)


VISION = {
    "siglip":  _tower("google/siglip-base-patch16-224",
                       norm=([0.5] * 3, [0.5] * 3), dim=768),
    "siglip2": _tower("google/siglip2-base-patch16-224",
                       norm=([0.5] * 3, [0.5] * 3), dim=768),
    "dinov2":  _tower("facebook/dinov2-base",
                       norm=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                       dim=768, cls=1),
    "clip":    _tower("openai/clip-vit-base-patch16",
                       norm=([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
                       dim=768, cls=1),
    "dinov2l": _tower("facebook/dinov2-large",
                       norm=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                       dim=1024, cls=1),
    "dinov2g": _tower("facebook/dinov2-giant",
                       norm=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                       dim=1536, cls=1),
    "siglip2so": _tower("google/siglip2-so400m-patch14-384",
                         norm=([0.5] * 3, [0.5] * 3), dim=1152, size=384),
}
TEXT = {
    "siglip_t": _tower("google/siglip-base-patch16-224", dim=768),
    "clip_t":   _tower("openai/clip-vit-base-patch16", dim=512),
    "minilm":   _tower("sentence-transformers/all-MiniLM-L6-v2", dim=384),
    "t5s":      _tower("google/flan-t5-small", dim=512),
    # mean-pooled T5 embeddings, L2-normalised and rescaled
    "t5sn":     _tower("google/flan-t5-small", dim=512, norm_scale=20.0),
    "t5bn":     _tower("google/flan-t5-base", dim=768, norm_scale=20.0),
}
_SUFFIX = re.compile(r"_(r?)p(\d+)$")


def _split_tag(tag):
    """'dinov2_p4_rp384' -> ('dinov2', 384); without `_rp<d>` the width is 0."""
    rand_proj = 0
    while match := _SUFFIX.search(tag):
        tag = tag[:match.start()]
        if match.group(1):
            rand_proj = int(match.group(2))
    return tag, rand_proj


def vis_dim(tags):
    """total embedding dim for a '+'-joined vision tag string"""
    return sum(VISION[t]["dim"] for t in tags.split("+"))


def load_vision(tag, pool=4, device="cuda"):
    """returns encode(u8 (B,S,S,3) np) -> (B, pool*pool, dim) float32 tensor on device.

    A tag may end in `_p<k>`, ignored because `pool` sets the grid, and in `_rp<d>`,
    which projects every token to d dims with a fixed seed-42 Gaussian matrix.
    """
    tag, rand_proj = _split_tag(tag)
    spec = VISION[tag]
    load_args = {"revision": spec["revision"], "local_files_only": True}
    if tag in ("siglip", "siglip2", "siglip2so"):
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(
            spec["hf"], dtype=torch.float16, **load_args
        ).to(device).eval()
    elif tag == "clip":
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(
            spec["hf"], dtype=torch.float16, **load_args
        ).to(device).eval()
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            spec["hf"], dtype=torch.float16, **load_args
        ).to(device).eval()
    mean = torch.tensor(spec["norm"][0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["norm"][1], device=device).view(1, 3, 1, 1)
    n_cls = spec.get("cls", 0)
    proj = None
    if rand_proj:  # seeded, so training and serving project identically
        g = torch.Generator().manual_seed(42)
        proj = (torch.randn(spec["dim"], rand_proj, generator=g) / spec["dim"] ** 0.5).half().to(device)
    in_size = spec.get("size", 224)

    def encode(u8):
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = Fn.interpolate(x, size=in_size, mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half()
        with torch.no_grad():
            h = model(pixel_values=x).last_hidden_state
            if n_cls:
                h = h[:, n_cls:]
            g = int(round(h.shape[1] ** 0.5))
            h = h.transpose(1, 2).reshape(-1, h.shape[-1], g, g)
            h = Fn.adaptive_avg_pool2d(h, pool).flatten(2).transpose(1, 2)
            if proj is not None:
                h = h.half() @ proj
        return h.float()

    return encode


def load_text(tag, device="cuda"):
    """returns encode([str,...]) -> (T, 1, dim) float32 tensor on device"""
    spec = TEXT[tag]
    from transformers import AutoTokenizer
    load_args = {"revision": spec["revision"], "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(spec["hf"], **load_args)
    if tag == "siglip_t":
        from transformers import SiglipTextModel
        model = SiglipTextModel.from_pretrained(
            spec["hf"], dtype=torch.float16, **load_args
        ).to(device).eval()

        def encode(strs):
            b = tok(strs, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                return model(input_ids=b["input_ids"].to(device)).pooler_output[:, None].float()
    elif tag == "clip_t":
        from transformers import CLIPTextModel
        model = CLIPTextModel.from_pretrained(
            spec["hf"], dtype=torch.float16, **load_args
        ).to(device).eval()

        def encode(strs):
            b = tok(strs, padding=True, truncation=True, max_length=77, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device))
                return out.pooler_output[:, None].float()
    else:  # mean-pooled encoders: minilm, t5s
        if tag in ("t5s", "t5sn", "t5bn"):
            from transformers import T5EncoderModel
            model = T5EncoderModel.from_pretrained(spec["hf"], **load_args).to(device).eval()
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(spec["hf"], **load_args).to(device).eval()

        def encode(strs):
            b = tok(strs, padding=True, truncation=True, max_length=64, return_tensors="pt")
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            with torch.no_grad():
                h = model(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask[..., None].float()
                return ((h * m).sum(1) / m.sum(1).clamp(min=1))[:, None].float()

    ns = spec.get("norm_scale", 0.0)
    if ns:
        base = encode

        def encode(strs):
            e = base(strs)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-8) * ns

    return encode
