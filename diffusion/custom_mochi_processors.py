#!/usr/bin/env python3
"""Custom attention processors for Mochi that force specific SDPA backends.

This allows true backend switching for benchmarking purposes.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional
from diffusers.models.attention_processor import MochiAttnProcessor2_0


class MochiAttnProcessorCUDNN(MochiAttnProcessor2_0):
    """Mochi attention processor forced to use cuDNN backend."""

    def __init__(self):
        super().__init__()
        self.backend = SDPBackend.CUDNN_ATTENTION
        self.backend_name = "CUDNN"

    def __call__(
        self,
        attn: "MochiAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        attn_outputs = []

        # FORCE CUDNN BACKEND HERE
        with sdpa_kernel([self.backend]):
            for idx in range(batch_size):
                mask = attention_mask[idx][None, :]
                valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

                valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
                valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
                valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

                valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
                valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
                valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

                attn_output = F.scaled_dot_product_attention(
                    valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
                )
                valid_sequence_length = attn_output.size(2)
                attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
                attn_outputs.append(attn_output)

        hidden_states = torch.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class MochiAttnProcessorFlash(MochiAttnProcessorCUDNN):
    """Mochi attention processor forced to use Flash Attention backend."""

    def __init__(self):
        super().__init__()
        self.backend = SDPBackend.FLASH_ATTENTION
        self.backend_name = "FLASH"


class MochiAttnProcessorEfficient(MochiAttnProcessorCUDNN):
    """Mochi attention processor forced to use Memory-Efficient backend."""

    def __init__(self):
        super().__init__()
        self.backend = SDPBackend.EFFICIENT_ATTENTION
        self.backend_name = "EFFICIENT"


class MochiAttnProcessorMath(MochiAttnProcessorCUDNN):
    """Mochi attention processor forced to use Math fallback backend."""

    def __init__(self):
        super().__init__()
        self.backend = SDPBackend.MATH
        self.backend_name = "MATH"


class MochiAttnProcessorFlashAttn2:
    """Mochi attention processor using direct Flash Attention 2 implementation."""

    def __init__(self):
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.backend_name = "FLASH_ATTN2_DIRECT"
        except ImportError:
            raise ImportError(
                "flash_attn is not installed. Install with: pip install flash-attn --no-build-isolation"
            )

    def __call__(
        self,
        attn: "MochiAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        # Flash Attention expects: (batch, seqlen, nheads, headdim)
        # Current shape: (batch, time, heads, headdim)
        # Need to transpose to: (batch, heads, time, headdim) then to (batch, time, heads, headdim)
        query = query.transpose(1, 2)  # (batch, heads, time, headdim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        encoder_query = encoder_query.transpose(1, 2)
        encoder_key = encoder_key.transpose(1, 2)
        encoder_value = encoder_value.transpose(1, 2)

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        attn_outputs = []

        for idx in range(batch_size):
            mask = attention_mask[idx][None, :]
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

            # Reshape for Flash Attention: (batch, seqlen, nheads, headdim)
            valid_query = valid_query.transpose(1, 2)  # (batch, seqlen, heads, headdim)
            valid_key = valid_key.transpose(1, 2)
            valid_value = valid_value.transpose(1, 2)

            # Use Flash Attention 2 directly
            attn_output = self.flash_attn_func(
                valid_query, valid_key, valid_value,
                dropout_p=0.0,
                causal=False,
            )

            # Reshape back: (batch, seqlen, heads, headdim) -> (batch, heads, seqlen, headdim)
            attn_output = attn_output.transpose(1, 2)

            valid_sequence_length = attn_output.size(2)
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)

        hidden_states = torch.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


# Mapping from backend names to processor classes
BACKEND_PROCESSORS = {
    "cudnn": MochiAttnProcessorCUDNN,
    "flash": MochiAttnProcessorFlash,
    "efficient": MochiAttnProcessorEfficient,
    "math": MochiAttnProcessorMath,
    "flash_attn2": MochiAttnProcessorFlashAttn2,
}


def set_mochi_attention_processor(model, backend: str):
    """Set a custom attention processor for all Mochi attention modules.

    Args:
        model: MochiTransformer3DModel instance
        backend: One of "cudnn", "flash", "efficient", "math", "flash_attn2"

    Returns:
        model: Modified model with new processors
    """
    if backend not in BACKEND_PROCESSORS:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {list(BACKEND_PROCESSORS.keys())}"
        )

    processor_class = BACKEND_PROCESSORS[backend]
    processor = processor_class()

    print(f"[INFO] Setting attention processor to: {processor.backend_name}")

    # Set processor on all attention modules
    for block in model.transformer_blocks:
        block.attn1.processor = processor

    # Also set on time_embed pooler if it exists
    if hasattr(model, "time_embed") and hasattr(model.time_embed, "pooler"):
        if hasattr(model.time_embed.pooler, "processor"):
            model.time_embed.pooler.processor = processor

    return model


def get_available_backends():
    """Get list of available backends on this system.

    Returns:
        list: Available backend names
    """
    available = []

    # Test SDPA backends
    test_tensor = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)

    for name in ["cudnn", "flash", "efficient", "math"]:
        try:
            processor_class = BACKEND_PROCESSORS[name]
            processor = processor_class()
            with sdpa_kernel([processor.backend]):
                _ = F.scaled_dot_product_attention(
                    test_tensor, test_tensor, test_tensor,
                    dropout_p=0.0, is_causal=False
                )
            available.append(name)
        except Exception:
            pass

    # Test Flash Attention 2
    try:
        from flash_attn import flash_attn_func
        available.append("flash_attn2")
    except ImportError:
        pass

    return available


if __name__ == "__main__":
    print("Custom Mochi Attention Processors")
    print("=" * 80)
    print("\nAvailable backends:")
    for backend in get_available_backends():
        print(f"  - {backend}")

    print("\nUsage:")
    print("""
    from custom_mochi_processors import set_mochi_attention_processor
    from diffusers import MochiTransformer3DModel

    # Load model
    model = MochiTransformer3DModel.from_pretrained(...)

    # Set backend
    model = set_mochi_attention_processor(model, "cudnn")
    # or "flash", "efficient", "math", "flash_attn2"
    """)
