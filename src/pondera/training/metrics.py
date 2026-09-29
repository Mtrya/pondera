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


def policy_loss(logits, targets, legal=None):
    if legal is None:
        return torch.nn.functional.cross_entropy(logits.float(), targets)
    return (
        legal_logits(logits, legal).logsumexp(-1)
        - logits.float().gather(1, targets[:, None]).squeeze(1)
    ).mean()


def factored_nll(factors, targets, tables):
    """Per-position teacher-move NLL under the factorized readout.

    tables = (MOVE_FROM, MOVE_TO, MOVE_PROMO) as device tensors indexed by
    move id; to/promo entries of -1 (claim_draw) mask their factors out.
    """
    from_logits, to_logits, promo_logits = factors
    move_from, move_to, move_promo = tables
    f = move_from[targets]
    t = move_to[targets]
    p = move_promo[targets]
    nll = -from_logits.float().log_softmax(-1).gather(1, f[:, None]).squeeze(1)
    has_move = (t >= 0).float()
    nll = nll - has_move * (
        to_logits.float().log_softmax(-1).gather(1, t.clamp_min(0)[:, None]).squeeze(1)
        + promo_logits.float()
        .log_softmax(-1)
        .gather(1, p.clamp_min(0)[:, None])
        .squeeze(1)
    )
    return nll


def factored_policy_loss(factors, targets, tables):
    return factored_nll(factors, targets, tables).mean()


class PolicyMetrics:
    """Retain per-position observations for slices, calibration and paired probes."""

    def __init__(self):
        self.batches = []

    @torch.no_grad()
    def update(self, logits, targets, legal, pieces, nll=None):
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
                # For factored readouts the headline per-move NLL comes from the
                # factor decomposition; raw_z on materialized log-probs then
                # measures the mass on structurally valid combinations.
                "loss": nll.detach() if nll is not None else raw_z - target,
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
                **(
                    {"factor_valid_mass": raw_z.exp()}
                    if nll is not None
                    else {}
                ),
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
            "factor_valid_mass",
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
def readout_changes(output, move_tokens=1):
    """Relative move-state changes between successive physical layer exits."""
    states = [
        readout.hidden[:, -1 - move_tokens : -1].flatten(1).float()
        for readout in output.readouts
    ]
    return {
        f"layer_{index}/move_relative_change": float(
            (
                (current - previous).norm(dim=-1)
                / previous.norm(dim=-1).clamp_min(1e-8)
            ).mean()
        )
        for index, (previous, current) in enumerate(zip(states, states[1:]), 2)
    }
