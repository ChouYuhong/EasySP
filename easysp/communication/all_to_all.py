
from typing import Any, Tuple
import torch
from torch import Tensor
import torch.distributed as dist

class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, scatter_dim: int, gather_dim: int, group: Any) -> Tensor:
        '''
        input: b l h d
        scatter_dim: along h (dim = -2 or 2)
        gather_dim: along l (dim = 1)
        group: should be sp_group
        '''
        
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group

        world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, world_size, scatter_dim)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        '''
        here we invert the gather_dim and scatter_dim
        '''
        return (SeqAllToAll.apply(*grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None, None)

'''
this is the communication operator with gradient
'''
def all2all(input, scatter_dim, gather_dim, sp_group):
    return SeqAllToAll.apply(input, scatter_dim, gather_dim, sp_group)