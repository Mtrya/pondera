"""The layer and execution layout shared by training and inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockConfig:
    attention: str
    group_end: bool = False
    navigation: bool = False
    value_source: str = "hidden"


@dataclass(frozen=True)
class SegmentConfig:
    repeat: int
    blocks: tuple[BlockConfig, ...]


DIMENSIONS = ("hidden_size", "intermediate_size", "num_heads", "num_kv_heads")
MODEL_SETTINGS = {*DIMENSIONS, "segments", "block_attn_res", "factorized_move_head"}


def _parse_segments(segments, block_attn_res):
    if not isinstance(segments, list) or not segments:
        raise ValueError("[model] segments must be a nonempty array of tables")
    parsed = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) - {"blocks", "repeat"}:
            raise ValueError("Each model segment takes blocks and an optional repeat")
        repeat = segment.get("repeat", 1)
        if type(repeat) is not int or repeat < 1:
            raise ValueError("Segment repeat must be a positive integer")
        if repeat > 1 and not block_attn_res:
            raise ValueError(
                "Repeated segments use Block AttnRes to read the prior pass"
            )
        blocks = segment.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("A model segment requires a nonempty blocks array")
        parsed.append(
            SegmentConfig(repeat, tuple(_parse_block(block) for block in blocks))
        )
    return tuple(parsed)


def _parse_block(block):
    if isinstance(block, str):
        block = {"attention": block}
    if (
        not isinstance(block, dict)
        or "attention" not in block
        or set(block) - {"attention", "group_end", "navigation", "value_source"}
    ):
        raise ValueError(
            "A block requires attention and accepts group_end, navigation, and value_source"
        )
    result = BlockConfig(**block)
    if result.attention not in ("kda", "softmax"):
        raise ValueError("Block attention must be kda or softmax")
    if type(result.group_end) is not bool or type(result.navigation) is not bool:
        raise ValueError("Block group_end and navigation must be booleans")
    if result.value_source not in ("hidden", "token_embedding"):
        raise ValueError("Block value_source must be hidden or token_embedding")
    if result.value_source == "token_embedding" and result.attention != "softmax":
        raise ValueError("token_embedding values require a softmax block")
    return result


def validate_model_config(config):
    if not isinstance(config, dict):
        raise ValueError("[model] must be a table")
    unknown = set(config) - MODEL_SETTINGS
    if unknown:
        raise ValueError(f"Unknown config key in [model]: {sorted(unknown)[0]}")
    for name in DIMENSIONS:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"[model] {name} must be a positive integer")
    if (
        config["hidden_size"] % config["num_heads"]
        or config["num_heads"] % config["num_kv_heads"]
    ):
        raise ValueError("[model] head counts must divide each other and hidden_size")
    for name in ("block_attn_res", "factorized_move_head"):
        if type(config.get(name, False)) is not bool:
            raise ValueError(f"[model] {name} must be a boolean")
    return _parse_segments(config.get("segments"), config.get("block_attn_res", False))
