# -*- coding: utf-8 -*-
"""
GLA to GLA-Ulysses Weight Migration Tool

This module provides utilities to migrate weights from a standard GLA model
to a GLA model with Ulysses sequence parallelism support.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torch.distributed import ProcessGroup
import copy
from collections import OrderedDict
from easysp.ulysses_layers.gla.configuration_gla import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM
import os
import torch.distributed as dist

class GLAWeightMigrator:
    
    @staticmethod
    def migrate_attention_layer(
        source_attn: nn.Module,
        target_attn: nn.Module,
        strict: bool = True
    ) -> Dict[str, Any]:
        migration_stats = {
            'matched_params': [],
            'missing_in_target': [],
            'missing_in_source': [],
            'shape_mismatches': []
        }
        
        source_state = source_attn.state_dict()
        target_state = target_attn.state_dict()
        
        for key, param in source_state.items():
            if key in target_state:
                if target_state[key].shape == param.shape:
                    target_state[key].copy_(param)
                    migration_stats['matched_params'].append(key)
                else:
                    migration_stats['shape_mismatches'].append({
                        'param': key,
                        'source_shape': param.shape,
                        'target_shape': target_state[key].shape
                    })
                    if not strict:
                        print(f"Warning: Shape mismatch for {key}, skipping...")
            else:
                migration_stats['missing_in_target'].append(key)
                if strict:
                    raise ValueError(f"Parameter {key} not found in target model")
        
        for key in target_state.keys():
            if key not in source_state:
                migration_stats['missing_in_source'].append(key)
                print(f"Info: New parameter {key} in Ulysses model, using initialized value")
        
        target_attn.load_state_dict(target_state, strict=False)
        
        return migration_stats
    
    @staticmethod
    def migrate_gla_block(
        source_block: nn.Module,
        target_block: nn.Module,
        layer_idx: int,
        config: Any
    ) -> Dict[str, Any]:
        block_stats = {
            'layer_idx': layer_idx,
            'attn_stats': None,
            'mlp_stats': None,
            'norm_stats': None
        }
        
        if hasattr(source_block, 'attn_norm') and hasattr(target_block, 'attn_norm'):
            target_block.attn_norm.load_state_dict(source_block.attn_norm.state_dict())
            block_stats['norm_stats'] = 'attn_norm migrated'
        
        if hasattr(source_block, 'mlp_norm') and hasattr(target_block, 'mlp_norm'):
            target_block.mlp_norm.load_state_dict(source_block.mlp_norm.state_dict())
            block_stats['norm_stats'] = 'mlp_norm migrated'
        
        if hasattr(source_block, 'attn') and hasattr(target_block, 'attn'):
            if source_block.attn.__class__.__name__ == 'GatedLinearAttention':
                block_stats['attn_stats'] = GLAWeightMigrator.migrate_attention_layer(
                    source_block.attn, 
                    target_block.attn,
                    strict=False
                )
            else:
                target_block.attn.load_state_dict(source_block.attn.state_dict())
                block_stats['attn_stats'] = 'Standard attention migrated'
        
        if hasattr(source_block, 'mlp') and hasattr(target_block, 'mlp'):
            target_block.mlp.load_state_dict(source_block.mlp.state_dict())
            block_stats['mlp_stats'] = 'MLP migrated'
        
        return block_stats


def create_ulysses_model_from_standard(
    source_model: nn.Module,
    config: Any,
    sp_group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a Ulysses-enabled model from a standard GLA model.
    
    Args:
        source_model: Source GLAForCausalLM model
        config: Model configuration
        sp_group: Process group for sequence parallelism
        device: Target device for the new model
        
    Returns:
        New model with Ulysses support and migrated weights
    """
    from fla.layers.gla import GatedLinearAttention
    from fla.models.gla.modeling_gla import GLAForCausalLM, GLABlock, GLAModel
    
    # Create a modified config for Ulysses
    ulysses_config = copy.deepcopy(config)
    
    # Create new model with Ulysses blocks
    class GLABlockUlysses(GLABlock):
        def __init__(self, config: Any, layer_idx: int):
            super().__init__(config, layer_idx)
            
            # Replace GatedLinearAttention with GatedLinearAttentionUlysses
            if hasattr(self, 'attn') and self.attn.__class__.__name__ == 'GatedLinearAttention':
                from easysp.ulysses_layers.gla import GatedLinearAttentionUlysses  # Import your Ulysses implementation
                
                self.attn = GatedLinearAttentionUlysses(
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
                    layer_idx=layer_idx,
                    sp_group=sp_group  # Add sequence parallel group
                )
    
    class GLAModelUlysses(GLAModel):
        def __init__(self, config: Any):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            
            self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList([
                GLABlockUlysses(config, layer_idx) 
                for layer_idx in range(config.num_hidden_layers)
            ])
            from fla.modules import RMSNorm
            self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
                config.hidden_size, 
                eps=config.norm_eps
            )
            self.gradient_checkpointing = False
    
    class GLAForCausalLMUlysses(GLAForCausalLM):
        def __init__(self, config: Any):
            super().__init__(config)
            self.model = GLAModelUlysses(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.config = config
            
    
    # Create new model
    target_model = GLAForCausalLMUlysses(config = ulysses_config)
    
    if device:
        target_model = target_model.to(device)
        source_model = source_model.to(device)
    
    # Migrate weights
    migration_report = migrate_model_weights(source_model, target_model, ulysses_config)
    
    return target_model, migration_report


def migrate_model_weights(
    source_model: nn.Module,
    target_model: nn.Module,
    config: Any
) -> Dict[str, Any]:
    """
    Migrate all weights from source to target model.
    
    Args:
        source_model: Source model with standard GLA
        target_model: Target model with Ulysses GLA
        config: Model configuration
        
    Returns:
        Complete migration report
    """
    migrator = GLAWeightMigrator()
    report = {
        'embeddings': None,
        'layers': [],
        'lm_head': None,
        'norm': None,
        'success': True
    }
    
    try:
        # Migrate embeddings
        if hasattr(source_model, 'model') and hasattr(target_model, 'model'):
            target_model.model.embeddings.load_state_dict(
                source_model.model.embeddings.state_dict()
            )
            report['embeddings'] = 'Successfully migrated'
            
            # Migrate each layer
            for idx, (source_layer, target_layer) in enumerate(
                zip(source_model.model.layers, target_model.model.layers)
            ):
                layer_report = migrator.migrate_gla_block(
                    source_layer, 
                    target_layer, 
                    idx, 
                    config
                )
                report['layers'].append(layer_report)
            
            # Migrate final norm
            target_model.model.norm.load_state_dict(
                source_model.model.norm.state_dict()
            )
            report['norm'] = 'Successfully migrated'
        
        # Migrate LM head
        if hasattr(source_model, 'lm_head') and hasattr(target_model, 'lm_head'):
            target_model.lm_head.load_state_dict(source_model.lm_head.state_dict())
            report['lm_head'] = 'Successfully migrated'
            
    except Exception as e:
        report['success'] = False
        report['error'] = str(e)
        print(f"Migration failed: {e}")
    
    return report


def validate_migration(
    source_model: nn.Module,
    target_model: nn.Module,
    local_rank: int = 0,
    sample_input: Optional[torch.Tensor] = None,
    tolerance: float = 1e-5
) -> bool:
    
    if sample_input is None:
        B, L, H, D = 1, 1024, 16, 128  
        sample_input = torch.randint(0, target_model.vocab_size, (B, L)).to('cuda')
    
    source_model.eval()
    target_model.eval()
    
    with torch.no_grad():
        source_output = source_model(sample_input)
        target_output = target_model(sample_input)
        
        if hasattr(source_output, 'logits') and hasattr(target_output, 'logits'):
            diff = torch.abs(source_output.logits - target_output.logits).max().item()
            if local_rank == 0:
                print(f"Maximum logit difference: {diff}")
            return diff < tolerance
        else:
            if local_rank == 0:
                print("Warning: Could not compare logits")
            return False
def init_sp_group(sp_size: int = 4):

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    assert world_size % sp_size == 0, f"world_size={world_size} 不能被 sp_size={sp_size} 整除"
    dp_size = world_size // sp_size

    dp_rank = rank // sp_size
    sp_rank_in_group = rank % sp_size

    group_ranks = [dp_rank * sp_size + i for i in range(sp_size)]
    sp_group = dist.new_group(ranks=group_ranks)

    if rank == 0:
        print(f"[Init] world_size={world_size}, dp_size={dp_size}, sp_size={sp_size}")
    print(f"[Rank {rank}] dp_rank={dp_rank}, sp_rank_in_group={sp_rank_in_group}, sp_group={group_ranks}")

    return sp_group, local_rank



def main():
    import torch
    
    model_path = "path/to/your/gla/model"

    config = GLAConfig(
                vocab_size=151936,
                hidden_size=2048,
                num_hidden_layers=24,
                num_heads=16,
                memory_efficient_checkpointing=False, 
                chunk_size=4096,
                use_cache=True,
                attn_mode='chunk',
            )
    source_model = GLAForCausalLM(config).to('cuda')

  
    sp_group, local_rank = init_sp_group(sp_size=4)    

    target_model, migration_report = create_ulysses_model_from_standard(
        source_model=source_model,
        config=config,
        sp_group=sp_group,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if local_rank == 0:
        print("Migration Report:")
        print(f"Embeddings: {migration_report['embeddings']}")
        print(f"LM Head: {migration_report['lm_head']}")
        print(f"Norm: {migration_report['norm']}")
        print(f"Success: {migration_report['success']}")
    
    # if local_rank == 0:
    is_valid = validate_migration(source_model, target_model, local_rank)
    if local_rank == 0:
        print("Validation successful!" if is_valid else "Validation failed!")
    
    if is_valid:
        pass
        torch.save(target_model.state_dict(), "gla_ulysses_model.pt")
        print("Model saved successfully!")
    
    return target_model


if __name__ == "__main__":
    main()