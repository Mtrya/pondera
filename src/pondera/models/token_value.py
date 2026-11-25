"""Use the original token embedding as the final softmax layer's value input."""

import torch
from torch.utils.checkpoint import checkpoint

from pondera.models.delta import KimiDeltaModel
from pondera.models.vanilla import CausalSelfAttention


class KimiDeltaTokenValueModel(KimiDeltaModel):
    architecture = "kda_token_value"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.blocks[-1].attention, CausalSelfAttention):
            raise ValueError("The final block must use softmax attention")
        self.register_buffer(
            "value_rms_scale",
            torch.ones(self.token_embedding.embedding_dim),
            persistent=False,
        )

    def _project(self, block, x, embedding):
        eps = block.norm_attention.eps
        if (
            embedding.device.type == "npu"
            and embedding.dtype == self.value_rms_scale.dtype == torch.float32
        ):
            import torch_npu

            token_input = torch_npu.npu_rms_norm(
                embedding, self.value_rms_scale, epsilon=eps
            )[0]
        else:
            mean_square = embedding.float().square().mean(dim=-1, keepdim=True)
            token_input = (embedding * torch.rsqrt(mean_square + eps)).to(
                embedding.dtype
            )
        return block.attention._project(
            block.norm_attention(x), value_input=token_input
        )

    @staticmethod
    def _last_ffn(block, x, attended):
        x = x + attended
        return x + block._ffn(block.norm_ffn(x))

    def _last_block(self, x, embedding, prepared):
        block = self.blocks[-1]
        q, k, v = self._project(block, x, embedding)
        attended = block.attention.attend_projected(q, k, v, prepared)
        return self._last_ffn(block, x, attended)

    def forward(self, input_ids, cu_seqlens, *, after_block=None):
        prepared = self.prepare(cu_seqlens, input_ids.device)
        x = embedding = self.token_embedding(input_ids)
        recompute = self.training and self.activation_checkpointing
        for index in range(len(self.blocks) - 1):
            block = self.blocks[index]
            metadata = prepared[type(block.attention)]
            if recompute:
                x = checkpoint(block, x, metadata, use_reentrant=False)
            else:
                x = block(x, metadata)
            if after_block is not None:
                x = after_block(index, x)
        args = (x, embedding, prepared[CausalSelfAttention])
        if recompute:
            x = checkpoint(self._last_block, *args, use_reentrant=False)
        else:
            x = self._last_block(*args)
        if after_block is not None:
            x = after_block(len(self.blocks) - 1, x)
        return self.head(self.norm(x))

    def forward_cached(self, input_ids, cu_seqlens, caches=None, *, after_block=None):
        x = embedding = self.token_embedding(input_ids)
        new_caches = []
        for index in range(len(self.blocks) - 1):
            block = self.blocks[index]
            old_cache = None if caches is None else caches[index]
            x, new_cache = block.forward_cached(x, cu_seqlens, old_cache)
            new_caches.append(new_cache)
            if after_block is not None:
                x = after_block(index, x)
        block = self.blocks[-1]
        old_cache = None if caches is None else caches[-1]
        q, k, v = self._project(block, x, embedding)
        attended, new_cache = block.attention.attend_projected_cached(
            q, k, v, cu_seqlens, old_cache
        )
        x = self._last_ffn(block, x, attended)
        if after_block is not None:
            x = after_block(len(self.blocks) - 1, x)
        return self.head(self.norm(x)), [*new_caches, new_cache]
