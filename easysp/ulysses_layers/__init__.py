from fla.layers import (
    Attention,
    HGRN2Attention,
    GatedLinearAttention,
)

from easysp.layers.dela import DecayLinearAttention

from .attn import AttentionUlysses
from .hgrn2 import HGRN2AttentionUlysses
from .gla import GatedLinearAttentionUlysses
from .dela import DecayLinearAttentionUlysses

def ulysseslize(model, sp_group):
    for layer in model.model.layers:
        module = layer.attn
        if isinstance(module, Attention):
            new_module = AttentionUlysses(
                hidden_size=module.hidden_size,
                num_heads=module.num_heads,
                num_kv_heads=module.num_kv_heads,
                head_dim=module.head_dim,
                qkv_bias=module.qkv_bias,
                qk_norm=module.qk_norm,
                window_size=module.window_size,
                rope_theta=module.rope_theta,
                max_position_embeddings=module.max_position_embeddings,
                layer_idx=module.layer_idx,
                use_rope=module.use_rope,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)

        elif isinstance(module, HGRN2Attention):
            new_module = HGRN2AttentionUlysses(
                mode=module.mode,
                hidden_size=module.hidden_size,
                num_heads=module.num_heads,
                expand_ratio=module.expand_ratio,
                use_short_conv=module.use_short_conv,
                conv_size=module.conv_size,
                conv_bias=module.conv_bias,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)

        elif isinstance(module, GatedLinearAttention):
            new_module = GatedLinearAttentionUlysses(
                mode=module.mode,
                hidden_size=module.hidden_size,
                expand_k=module.expand_k,
                expand_v=module.expand_v,
                num_heads=module.num_heads,
                num_kv_heads=module.num_kv_heads,
                use_short_conv=module.use_short_conv,
                conv_size=module.conv_size,
                conv_bias=module.conv_bias,
                use_output_gate=module.use_output_gate,
                gate_fn=module.gate_fn,
                gate_logit_normalizer=module.gate_logit_normalizer,
                clamp_min=module.clamp_min,
                fuse_norm=False,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)
        elif isinstance(module, DecayLinearAttention):
            new_module = DecayLinearAttentionUlysses(
                hidden_size=module.hidden_size,
                expand_qk=module.expand_qk,
                num_heads=module.num_heads,
                feature_map=module.feature_map,
                use_short_conv=module.use_short_conv,
                conv_size=module.conv_size,
                conv_bias=module.conv_bias,
                use_output_gate=module.use_output_gate,
                clamp_min=module.clamp_min,
                fuse_norm=False,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)

        else:
            NotImplementedError(f"Unsupported module {module}")
