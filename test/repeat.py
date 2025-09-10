import torch
from einops import rearrange, repeat
import pdb

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


if __name__ == "__main__":
    batch, num_key_value_heads, slen, head_dim = 2, 8, 512, 128
    num_group = 4

    hidden_states_ref = torch.randn(batch, num_key_value_heads, slen, head_dim)
    hidden_states_ein = hidden_states_ref.clone().detach().transpose(1, 2).contiguous()

    repeat_ref = repeat_kv(hidden_states_ref, num_group)

    repeat_ein = repeat(hidden_states_ein, "b l h d -> b (h g) l d", g=num_group)
    diff = repeat_ein - repeat_ref
    pdb.set_trace()