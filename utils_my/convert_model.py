import torch
from copy import deepcopy
maskformerPath ="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify/model_final.pth"
maskformerNoClipPath = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify/model_final_noclip.pth"

maskformerModel = torch.load(maskformerPath)
maskformerModelC = deepcopy(maskformerModel)
# import pdb; pdb.set_trace()

for key, value in maskformerModelC["model"].items():
    
    if key.startswith("clip_adapter"):
        # import pdb; pdb.set_trace()
        maskformerModel['model'].pop(key)
# import pdb; pdb.set_trace()
torch.save(maskformerModel, maskformerNoClipPath)