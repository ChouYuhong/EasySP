import torch
import pdb
from easysp.models.dela import DELAForCausalLM
from easysp.models.dela import DELAConfig

def reinit_weights(model):
    for module in model.modules():
        if hasattr(module, '_is_hf_initialized'):
            module._is_hf_initialized = False
    model.init_weights()

if __name__ == "__main__":
    config = DELAConfig.from_pretrained("/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/codebase/long_context_team/ULTra/main/model_config/1B3_baseline/hybrid_dela")
    with torch.device("meta"):
        model = DELAForCausalLM(config)
    model.to_empty(device="cuda")
    reinit_weights(model)
    inputs = torch.randint(0, 32000, (1, 8 * 2048)).to(torch.long).cuda()
    model = model.to(torch.bfloat16)
    pdb.set_trace()
    output = model(input_ids=inputs, labels=inputs)
    