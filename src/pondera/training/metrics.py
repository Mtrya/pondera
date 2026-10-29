"""Policy diagnostics separating legal-move preference from legality learning."""

import numpy as np
import torch


def legal_logits(logits, legal):
    """Gather precomputed legal actions; negative indices are padding."""
    return (
        logits.float()
        .gather(1, legal.clamp_min(0).long())
        .masked_fill(legal < 0, float("-inf"))
    )


class PolicyMetrics:
    """Retain per-position observations for slices, calibration and paired probes."""

    def __init__(self):
        self.batches = []

    @torch.no_grad()
    def update(self, logits, targets, legal, pieces):
        logits = logits.float()
        raw_z = logits.logsumexp(-1)
        restricted = legal_logits(logits, legal)
        legal_z = restricted.logsumexp(-1)
        target = logits.gather(1, targets[:, None]).squeeze(1)
        prediction = logits.argmax(-1)
        legal_prediction = legal.gather(1, restricted.argmax(-1)[:, None]).squeeze(1)
        target_slot = (legal == targets[:, None]).long().argmax(-1)
        slots = torch.arange(legal.shape[1], device=legal.device)
        rank = (
            1
            + (restricted > target[:, None]).sum(-1)
            + ((restricted == target[:, None]) & (slots < target_slot[:, None])).sum(-1)
        )
        self.batches.append(
            {
                "loss": raw_z - target,
                "legal_loss": legal_z - target,
                "legality_penalty": raw_z - legal_z,
                "legal_mass": (legal_z - raw_z).exp(),
                "top1": (prediction == targets).float(),
                "legal_top1": (legal_prediction == targets).float(),
                "legal_top5": (rank <= 5).float(),
                "legal_mrr": rank.float().reciprocal(),
                "legal_rank": rank,
                "illegal_argmax": (~(legal == prediction[:, None]).any(-1)).float(),
                "confidence": (logits.max(-1).values - raw_z).exp(),
                "legal_confidence": (restricted.max(-1).values - legal_z).exp(),
                "entropy": (logits.softmax(-1) * (raw_z[:, None] - logits)).sum(-1),
                "pieces": pieces,
                "target": targets,
                "prediction": prediction,
                "legal_prediction": legal_prediction,
                "legal_count": (legal >= 0).sum(-1),
            }
        )

    def result(self, artifact=None):
        arrays = {
            key: torch.cat([batch[key] for batch in self.batches]).cpu().numpy()
            for key in self.batches[0]
        }
        if artifact is not None:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, **arrays)
        names = (
            "loss",
            "legal_loss",
            "legality_penalty",
            "legal_mass",
            "top1",
            "legal_top1",
            "legal_top5",
            "legal_mrr",
            "illegal_argmax",
            "confidence",
            "legal_confidence",
            "entropy",
        )
        result = {
            name: float(arrays[name].mean()) for name in names if name in arrays
        }
        result["positions"] = len(arrays["loss"])
        for prefix in ("", "legal_"):
            confidence = arrays[f"{prefix}confidence"]
            correct = arrays[f"{prefix}top1"]
            bins = np.minimum((confidence * 10).astype(int), 9)
            result[f"{prefix}ece"] = sum(
                float(abs((confidence[bins == i] - correct[bins == i]).sum()))
                for i in range(10)
            ) / len(confidence)
        for lower, upper in ((2, 10), (11, 20), (21, 32)):
            mask = (arrays["pieces"] >= lower) & (arrays["pieces"] <= upper)
            prefix = f"pieces_{lower}_{upper}/"
            result[prefix + "positions"] = int(mask.sum())
            if mask.any():
                result.update(
                    {
                        prefix + name: float(arrays[name][mask].mean())
                        for name in names
                        if name in arrays
                    }
                )
        return result


@torch.no_grad()
def readout_changes(output):
    """Relative move-state changes between successive physical layer exits."""
    states = [readout.hidden[:, -2].float() for readout in output.readouts]
    return {
        f"layer_{index}/move_relative_change": float(
            (
                (current - previous).norm(dim=-1)
                / previous.norm(dim=-1).clamp_min(1e-8)
            ).mean()
        )
        for index, (previous, current) in enumerate(zip(states, states[1:]), 2)
    }
