"""Internal-health probes: residual energy, attention, dead units, rank, drift.

The probe observes one fixed evaluation batch per call. It answers health
questions (instability, dead pathways, starving embeddings); it never ranks
branches. Training-path overhead is one attribute check per layer.
"""

import numpy as np
import torch


def _blocks(model):
    return [layer for core in model.backbone.cores for layer in core.layers]


def _embedding_weight(model):
    """Token embedding table of either board-policy architecture."""
    if hasattr(model, "token_embedding"):
        return model.token_embedding.weight
    return model.board_embedding.tokens.weight


def effective_rank(singular_values):
    """Entropy effective rank from singular values."""
    total = singular_values.sum().clamp_min(1e-12)
    probs = singular_values / total
    return float(torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum()))


def _group(name):
    if name.startswith("backbone.cores."):
        parts = name.split(".")
        return f"core{parts[2]}_layer_{int(parts[4]):02d}"
    if name.startswith(("board_embedding", "token_embedding")):
        return "embedding"
    return name.split(".")[0]


class InternalProbe:
    """Attach once per training run; collect at validation points."""

    def __init__(self, model, init_file=None):
        self.active = False
        self.ffn_inputs = {}
        self.handles = [
            block.ffn.down_proj.register_forward_pre_hook(self._ffn_hook(index))
            for index, block in enumerate(_blocks(model))
        ]
        embedding = _embedding_weight(model)
        weights = {"tokens": embedding}
        if init_file is not None and init_file.exists():
            loaded = np.load(init_file)
            self.initial = {name: torch.from_numpy(loaded[name]) for name in weights}
        else:
            self.initial = {
                name: weight.detach().float().cpu().clone()
                for name, weight in weights.items()
            }
            if init_file is not None:
                init_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    init_file,
                    **{name: weight.numpy() for name, weight in self.initial.items()},
                )
        self.previous = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
        }

    def _ffn_hook(self, index):
        def hook(module, inputs):
            if self.active:
                (gated,) = inputs
                self.ffn_inputs[index] = gated.detach()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def collect(self, model, inputs):
        """One no-grad capture pass over the probe batch; returns (scalars, arrays)."""
        blocks = _blocks(model)
        self.ffn_inputs = {}
        self.active = True
        for block in blocks:
            block.attention_capture = []
        was_training = model.training
        model.eval()
        output = model(**inputs, return_backbone=True, capture=True)[-1]
        if was_training:
            model.train()
        self.active = False
        captures = []
        for block in blocks:
            captures.append(block.attention_capture)
            block.attention_capture = None

        scalars, arrays = {}, {}
        readouts = sorted(output.readouts, key=lambda r: (r.core, r.layer))
        for position, readout in enumerate(readouts):
            hidden = readout.hidden[readout.active].float()
            scalars[f"layer_{position:02d}/residual_rms"] = float(
                hidden.pow(2).mean().sqrt()
            )
            spectrum = torch.linalg.svdvals(hidden[:4096])
            scalars[f"layer_{position:02d}/effective_rank"] = effective_rank(spectrum)
            arrays[f"rank_spectrum_{position:02d}"] = spectrum.cpu().numpy()
        del output

        heads = blocks[0].heads
        entropy = np.full((len(blocks), heads), np.nan, dtype=np.float32)
        peak = np.full((len(blocks), heads), np.nan, dtype=np.float32)
        for index, captured in enumerate(captures):
            if not captured:
                continue
            for head, (head_entropy, head_peak) in enumerate(captured[0]):
                entropy[index, head] = head_entropy
                peak[index, head] = head_peak
        arrays["attention_entropy"] = entropy
        arrays["attention_peak"] = peak
        for index in range(len(blocks)):
            scalars[f"layer_{index:02d}/attention_entropy_mean"] = float(
                np.nanmean(entropy[index])
            )
            scalars[f"layer_{index:02d}/attention_entropy_min"] = float(
                np.nanmin(entropy[index])
            )
            scalars[f"layer_{index:02d}/attention_peak_mean"] = float(
                np.nanmean(peak[index])
            )
            scalars[f"layer_{index:02d}/attention_peak_max"] = float(
                np.nanmax(peak[index])
            )

        for index, gated in sorted(self.ffn_inputs.items()):
            gated = gated.float()
            # Every readout shares the execution's token mask; trace padding is
            # excluded so dead-channel counts reflect real tokens only.
            if readouts and readouts[0].active.shape == gated.shape[:2]:
                channel_max = gated[readouts[0].active].abs().amax(dim=0).cpu().numpy()
            else:
                channel_max = gated.abs().amax(dim=(0, 1)).cpu().numpy()
            arrays[f"ffn_channel_max_{index:02d}"] = channel_max
            scalars[f"layer_{index:02d}/ffn_dead_fraction"] = float(
                (channel_max < 1e-4).mean()
            )
            for quantile in (50, 90, 99):
                scalars[f"layer_{index:02d}/ffn_channel_max_p{quantile}"] = float(
                    np.percentile(channel_max, quantile)
                )

        weight = _embedding_weight(model)
        current = weight.detach().float().cpu()
        initial = self.initial["tokens"]
        norms = current.norm(dim=1)
        scalars["embedding/tokens_row_norm_mean"] = float(norms.mean())
        scalars["embedding/tokens_row_norm_std"] = float(norms.std())
        delta = (current - initial).norm(dim=1) / initial.norm(dim=1).clamp_min(1e-8)
        arrays["embedding_tokens_relative_delta"] = delta.numpy()
        scalars["embedding/tokens_starved_fraction"] = float(
            (delta < 0.02).float().mean()
        )

        updates = {}
        for name, parameter in model.named_parameters():
            current = parameter.detach().float().cpu().clone()
            previous = self.previous[name]
            group = _group(name)
            delta, base = updates.get(group, (0.0, 0.0))
            updates[group] = (
                delta + float((current - previous).pow(2).sum()),
                base + float(previous.pow(2).sum()),
            )
            self.previous[name] = current
        for group, (delta, base) in updates.items():
            scalars[f"drift/{group}"] = delta**0.5 / max(base**0.5, 1e-12)

        return scalars, arrays
