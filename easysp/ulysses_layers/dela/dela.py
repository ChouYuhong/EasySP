# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributed as dist
from torch.distributed import ProcessGroup
from easysp.communication import all2all

from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

# @torch.compile
def mamba_decay(A_log, alpha, dt_bias):
    g = -A_log.float().exp() * F.softplus(alpha.float() + dt_bias)
    return g.contiguous()

class DecayLinearAttentionUlysses(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_qk (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_qk: float = 0.5,
        num_heads: int = 4,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        sp_group: ProcessGroup = None,
    ) -> DecayLinearAttentionUlysses:
        super().__init__()

        # NOTE general hyperparameters
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_qk = expand_qk
        self.num_heads = num_heads
        self.feature_map = feature_map
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        # NOTE optional hyperparameters
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        
        # NOTE recommended hyperparameters
        self.use_output_gate = use_output_gate

        self.qk_dim = int(hidden_size * expand_qk)
        self.value_dim = hidden_size
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        self.sp_group = sp_group

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{mode}`."
        assert self.qk_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.qk_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # NOTE initialize the decay related factors
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.qk_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.qk_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormGated(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps
            )
            self.gate_fn = ACT2FN[gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # NOTE here we compute the decay
        alpha = self.a_proj(hidden_states)
        gk = mamba_decay(self.A_log, alpha, self.dt_bias).view(batch_size, q_len, self.num_heads, 1).repeat(1, 1, 1, self.head_qk_dim)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_qk_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_qk_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            q, k, v, gk = [all2all(input=x, scatter_dim=2, gather_dim=1, sp_group=self.sp_group) for x in (q, k, v, gk)]
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            o = all2all(input=o, scatter_dim=1, gather_dim=2, sp_group=self.sp_group)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values