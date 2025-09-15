# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'  # 设置可见的GPU设备
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.gla import GatedLinearAttention
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as GLAMLP
from fla.modules import RMSNorm
from datetime import datetime

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)

def memory_efficient_model_forward(
    model,
    input_ids: torch.Tensor,
    init_state: torch.Tensor,
    chunk_size: int = 8192,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = False,
    **kwargs
):

    return CheckpointedModelFunction.apply(
        model, input_ids, init_state, chunk_size, attention_mask, past_key_values, use_cache, kwargs
    )
import torch
from typing import Optional, List, Tuple, Union

class CheckpointedModelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                model,
                input_ids: torch.Tensor,
                init_state: torch.Tensor,
                chunk_size: int,
                attention_mask: Optional[torch.Tensor],
                past_key_values,
                use_cache: bool,
                kwargs: dict):

        B, L = input_ids.shape
        num_layers = len(model.layers)
        print("process {} chunks, chunk_size: {}".format(L // chunk_size + 1, chunk_size))
        chunks: List[Tuple[int, int]] = []
        s = 0
        while s < L:
            e = min(s + chunk_size, L)
            chunks.append((s, e))
            s = e

        ctx.model = model
        ctx.kwargs = kwargs
        ctx.use_cache = use_cache
        ctx.chunks = chunks
        ctx.num_layers = num_layers

        mask_saved = attention_mask if attention_mask is not None else torch.tensor([], device=input_ids.device)
        ctx.save_for_backward(input_ids, mask_saved)

        current_states: List[Optional[torch.Tensor]] = [init_state.detach().clone() for _ in range(num_layers)]
        state_list: List[List[Optional[torch.Tensor]]] = []  # state_list[t][l] = 进入chunk t 前第 l 层的 state
        outputs_per_chunk: List[torch.Tensor] = []

        with torch.no_grad():
            for (start, end) in chunks:
                state_list.append([s.detach().clone() if s is not None else None for s in current_states])

                input_chunk = input_ids[:, start:end]
                mask_chunk = None if attention_mask is None else attention_mask[:, start:end]

                hidden_states = model.embeddings(input_chunk)

                new_states: List[Optional[torch.Tensor]] = []
                for layer_idx, layer in enumerate(model.layers):
                    out = layer(
                        hidden_states=hidden_states,
                        attention_mask=mask_chunk,
                        past_key_values=None,
                        use_cache=use_cache,
                        recurrent_state=current_states[layer_idx],
                        **kwargs
                    )
                    hidden_states, _, _, layer_state, *rest = out
                    new_states.append(layer_state)

                outputs_per_chunk.append(hidden_states)
                current_states = new_states

        output = torch.cat(outputs_per_chunk, dim=1)  # [B, L, hidden]
        output = model.norm(output)

        ctx.state_list = state_list

        final_states = tuple(current_states)  # (state_l0, state_l1, ..., state_l{L-1})
        return output, past_key_values, final_states

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor,
                 grad_past_kv,
                 grad_final_states: Optional[Union[List[Optional[torch.Tensor]], Tuple[Optional[torch.Tensor], ...], torch.Tensor]]):

        input_ids, mask_saved = ctx.saved_tensors
        model = ctx.model
        kwargs = ctx.kwargs
        use_cache = ctx.use_cache
        chunks = ctx.chunks
        num_layers = ctx.num_layers
        state_list = ctx.state_list

        attention_mask = None if mask_saved.numel() == 0 else mask_saved

        if isinstance(grad_final_states, (list, tuple)):
            assert len(grad_final_states) == num_layers, \
                f"grad_final_states 长度 {len(grad_final_states)} 与模型层数 {num_layers} 不一致"
            S_grads: List[Optional[torch.Tensor]] = [
                (g.detach() if g is not None else None) for g in grad_final_states
            ]
        elif grad_final_states is None:
            S_grads = [None for _ in range(num_layers)]
        else:
            S_grads = [None for _ in range(num_layers - 1)] + [grad_final_states.detach()]

        for t in reversed(range(len(chunks))):
            start, end = chunks[t]
            grad_chunk = grad_output[:, start:end, :]  

            prev_states = [
                (s.detach().clone().requires_grad_(True) if s is not None else None)
                for s in state_list[t]
            ]

            with torch.enable_grad():
                input_chunk = input_ids[:, start:end]
                mask_chunk = None if attention_mask is None else attention_mask[:, start:end]

                hidden_states = model.embeddings(input_chunk)
                new_states: List[Optional[torch.Tensor]] = []
                for layer_idx, layer in enumerate(model.layers):
                    out = layer(
                        hidden_states=hidden_states,
                        attention_mask=mask_chunk,
                        past_key_values=None,
                        use_cache=use_cache,
                        recurrent_state=prev_states[layer_idx],
                        **kwargs
                    )
                    hidden_states, _, _, layer_state, *rest = out
                    new_states.append(layer_state)

                hidden_states = model.norm(hidden_states)

                outs = [hidden_states]
                grads = [grad_chunk]
                for s_final, s_grad in zip(new_states, S_grads):
                    if s_grad is not None:
                        outs.append(s_final)
                        grads.append(s_grad)

                torch.autograd.backward(tuple(outs), tuple(grads))

                S_grads = [
                    (s.grad.detach() if (s is not None and s.grad is not None) else None)
                    for s in prev_states
                ]

        return (None,  # model
                None,  # input_ids
                None,  # init_state
                None,  # chunk_size
                None,  # attention_mask
                None,  # past_key_values
                None,  # use_cache
                None)  # kwargs

class GLABlock(nn.Module):
    def __init__(self, config: GLAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        recurrent_state: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # print("attn input hidden state requires_grad:", hidden_states.requires_grad)
        # print("attn input hidden state grad_fn:", hidden_states.grad_fn)
        hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values, recurrent_state = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            S0=recurrent_state,
            **kwargs
        )
        # print("attn output hidden state requires_grad:", hidden_states.requires_grad)
        # print("attn output hidden state grad_fn:", hidden_states.grad_fn)
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, recurrent_state)
        return outputs


class GLAPreTrainedModel(PreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")
class GLAModel(GLAPreTrainedModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        
        # Memory-efficient checkpointing settings
        # self.memory_efficient_checkpointing = getattr(config, 'memory_efficient_checkpointing', False)
        self.memory_efficient_checkpointing = True
        self.chunk_size = getattr(config, 'chunk_size', 8192)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def enable_memory_efficient_checkpointing(self, chunk_size: int = 8192):
        """Enable memory-efficient checkpointing."""
        self.memory_efficient_checkpointing = True
        self.chunk_size = chunk_size
        logger.info(f"Enabled memory-efficient checkpointing with chunk_size={chunk_size}")

    def disable_memory_efficient_checkpointing(self):
        """Disable memory-efficient checkpointing."""
        self.memory_efficient_checkpointing = False
        logger.info("Disabled memory-efficient checkpointing")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_heads: Optional[int] = 16,
        dk: Optional[int] = 64,
        dv: Optional[int] = 128,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`GLAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values) if past_key_values is not None else None

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if (self.memory_efficient_checkpointing 
            # and 
            # self.training and 
            # input_ids is not None and 
            # input_ids.shape[1] > self.chunk_size
            ):
            
            logger.info(f"Using memory-efficient checkpointing for sequence length {input_ids.shape[1]}")
            init_state = torch.zeros((input_ids.shape[0], num_heads, dk, dv), device=input_ids.device, requires_grad=True)
            
            return memory_efficient_model_forward(
                model=self,
                input_ids=input_ids,
                init_state=init_state,
                chunk_size=self.chunk_size,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        else:
            if inputs_embeds is None:
                print(input_ids.shape, "input_ids shape")
                inputs_embeds = self.embeddings(input_ids)
            hidden_states = inputs_embeds

            all_hidden_states = () if output_hidden_states else None
            all_attns = () if output_attentions else None
            
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        past_key_values,
                        use_cache,
                        output_attentions,
                        **kwargs
                    )
                else:
                    hidden_states, attentions, past_key_values,layer_state = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        **kwargs
                    )
                if output_attentions:
                    all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Handle return format
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attns,
        # ), layer_state
        return hidden_states, None, layer_state



class GLAForCausalLM(GLAPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def enable_memory_efficient_checkpointing(self, chunk_size: int = 8192):
        """Enable memory-efficient checkpointing."""
        self.model.enable_memory_efficient_checkpointing(chunk_size)

    def disable_memory_efficient_checkpointing(self):
        """Disable memory-efficient checkpointing."""
        self.model.disable_memory_efficient_checkpointing()

    def generate(self, *args, **kwargs):
        # Temporarily disable memory-efficient checkpointing during generation
        was_enabled = self.model.memory_efficient_checkpointing
        if was_enabled:
            self.model.disable_memory_efficient_checkpointing()
        
        try:
            result = super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception
        finally:
            # Re-enable if it was enabled before
            if was_enabled:
                self.model.enable_memory_efficient_checkpointing()
        
        return result

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = None, # ori is 0
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, _, layer_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        print("GLAForCausalLM hidden_states shape:", hidden_states.shape)
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training and labels is not None

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        ), layer_state
def set_seed(seed: int = 42):
    print(f"Setting random seed to {seed}")
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("✅ All random seeds set successfully")


def print_autograd_graph(output_tensor_or_fn, indent=0):
    if isinstance(output_tensor_or_fn, torch.Tensor):
        fn = output_tensor_or_fn.grad_fn
        if fn is None:
            print(" " * indent + f"Leaf Tensor: no grad_fn")
            return
    else:
        fn = output_tensor_or_fn

    print(" " * indent + f"{type(fn).__name__}")

    if hasattr(fn, "next_functions"):
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                print_autograd_graph(next_fn, indent + 2)
            else:
                print(" " * (indent + 2) + "None (leaf or constant)")

def acc_test(seed: int = 42):
    
    from fla.models.gla.configuration_gla import GLAConfig
    
    config = GLAConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_heads=16,
        memory_efficient_checkpointing=False, 
        chunk_size=512,
        use_cache=True,
        attn_mode='chunk',
    )
    
    batch_size, seq_len = 1, 1024*8
    
    set_seed(seed)
    
    base_model = GLAModel(config).cuda()
    base_model.train()
    
    model_state = base_model.state_dict()
    
    torch.manual_seed(seed)
    base_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    

    print(f"\n---标准forward ---")
    model1 = GLAModel(config).cuda()
    model1.load_state_dict(model_state)  
    model1.train()
    model1.disable_memory_efficient_checkpointing()
    

    input_ids1 = base_input_ids.clone()
    
    torch.manual_seed(seed + 1) 

    outputs1, _, layer_state1 = model1(input_ids1)

    loss1 = outputs1.sum()
    
    print(f"标准forward - Loss: {loss1.item()}")
    
    loss1.backward()
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"/home/zhliu/profiler/visual_mem_{timestamp}.pickle"
    
    grads1 = {}
    for name, param in model1.named_parameters():
        if param.grad is not None:
            grads1[name] = param.grad.clone()
    
    print(f"标准forward - 有梯度的参数数量: {len(grads1)}")
    
    print(f"\nMemory-efficient checkpointing ---")
    
    model2 = GLAModel(config).cuda()
    model2.load_state_dict(model_state)  
    model2.train()
    # model2.disable_memory_efficient_checkpointing()  # 
    model2.enable_memory_efficient_checkpointing(chunk_size=2048)  
    
    input_ids2 = base_input_ids.clone()
    

    torch.manual_seed(seed + 1)  

    outputs2, _, layer_state2 = model2(input_ids2)
    # print("q diff:", (q1 - q2).abs().max().item())
    # print("k diff:", (k1 - k2).abs().max().item())
    # print("v diff:", (v1 - v2).abs().max().item())
    # print("gk diff:", (gk1 - gk2).abs().max().item())
    print("layer_state diff:", (layer_state1[-1] - layer_state2[-1]).abs().max().item()/layer_state1[-1].abs().max().item())
    print("outputs diff:", (outputs1 - outputs2).abs().max().item())
    loss2 = outputs2.sum()

    print(f"Checkpointing forward - Loss: {loss2.item()}")
    
    loss2.backward()
    
    grads2 = {}
    for name, param in model2.named_parameters():
        # print(name, param.shape, True if param.grad is not None else False)
        if param.grad is not None:
            grads2[name] = param.grad.clone()
    
    print(f"Checkpointing forward - 有梯度的参数数量: {len(grads2)}")
    
    base_model.train()  
    grads0 = {}
    for name, param in base_model.named_parameters():
        if param.grad is not None:
            grads0[name] = param.grad.clone()
    print(f"基准模型 - 有梯度的参数数量: {len(grads0)}")

    # ========== 比较结果 ==========
    print(f"\n--- 精度比较 ---")
    # 比较layer state
    print(f"layer_state1[-1] shape: {layer_state1[-1].shape}")
    print(f"layer_state2[-1] shape: {layer_state2[-1].shape}")

    # state_diff = (layer_state1 - layer_state2).abs().max().item()
    # print(f"Layer state差异: {state_diff:.2e}") 
    # 比较loss

    output_diff = (outputs1 - outputs2).abs().max().item()
    print(f"output abs diff: {output_diff:.2e}")
    
    
    max_abs_diff = 0
    max_rel_diff = 0
    total_abs_diff = 0
    total_params = 0
    
    print(f"\ngrad:")
    for i, name in enumerate(list(grads1.keys())):
        if name in grads2:
            abs_diff = (grads1[name] - grads2[name]).abs().max().item()
            grad_norm = grads1[name].abs().max().item()
            rel_diff = abs_diff / max(grad_norm, 1e-12)
            
            print(f"  {name}: 绝对差异={abs_diff:.2e}, 相对差异={rel_diff:.6f}")
            
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            total_abs_diff += abs_diff
            total_params += 1
    
if __name__ == "__main__":
    # debug_model()
    acc_test()
    # test_ckpt_1B()