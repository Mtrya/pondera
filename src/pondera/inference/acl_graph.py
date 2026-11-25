"""Request-owned Ascend graphs for the fixed-shape segments of KDA decode."""

import torch
import torch_npu
from torch.utils._pytree import tree_leaves, tree_map


def _copy_caches(target, source):
    for dest, value in zip(tree_leaves(target), tree_leaves(source), strict=True):
        dest.copy_(value)


class _Segment:
    def __init__(self, blocks, hidden, caches):
        self.blocks = blocks
        self.input = hidden.clone()
        self.caches = tree_map(torch.clone, caches)
        self.offsets = torch.tensor([0, 1], dtype=torch.int32)
        self.graph = torch_npu.npu.NPUGraph()
        self.stream = torch_npu.npu.Stream()

        current = torch_npu.npu.current_stream()
        self.stream.wait_stream(current)
        with torch_npu.npu.stream(self.stream):
            for _ in range(3):
                self._forward()
        current.wait_stream(self.stream)
        with torch_npu.npu.graph(self.graph, stream=self.stream):
            self.output = self._forward()

    def _forward(self):
        x = self.input
        for block, cache in zip(self.blocks, self.caches, strict=True):
            x, updated = block.forward_cached(x, self.offsets, cache)
            # Replays read the original tensor addresses, with the last token's data.
            _copy_caches(cache, updated)
        return x

    def step(self, x):
        self.input.copy_(x)
        self.graph.replay()
        return self.output


class KdaDecodeGraph:
    """Capture the two KDA segments while executing growing softmax caches eagerly."""

    def __init__(self, model, token, caches):
        self.model = model
        hidden = model.token_embedding(token)
        self.segments = (
            _Segment(model.blocks[:3], hidden, caches[:3]),
            _Segment(model.blocks[4:7], hidden, caches[4:7]),
        )
        self.reset(caches)

    def reset(self, caches):
        _copy_caches(self.segments[0].caches, caches[:3])
        _copy_caches(self.segments[1].caches, caches[4:7])

    def step(self, token, offsets, caches):
        x = self.model.token_embedding(token)
        updated = []
        for index, segment in zip((3, 7), self.segments, strict=True):
            x = segment.step(x)
            x, cache = self.model.blocks[index].forward_cached(
                x, offsets, caches[index]
            )
            updated.extend((*segment.caches, cache))
        return self.model.head(self.model.norm(x)), updated
