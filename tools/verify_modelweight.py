import torch


def load_model(model_path):
    model = torch.load(model_path)
    
    model_weight = model["model"]
    
    return model_weight

if __name__=="__main__":

    path_nolearn_pos = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output1229/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt_diceloss_img384_ignore157_bat32_clipgrad/model_final.pth"
    path_learn_pos = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output1229/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt_diceloss_bcefocal_alllayerlearnposition_img384_bat32/model_final.pth"
    path_learn_pos_5000 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output1229/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt_diceloss_bcefocal_alllayerlearnposition_img384_bat32/model_0004999.pth"

    model_weight_no = load_model(path_nolearn_pos)
    model_weight = load_model(path_learn_pos)
    model_weight_5000 = load_model(path_learn_pos_5000)
    nameList = []

    for name, param in model_weight.items():
        # if name.startswith("clip_adapter.clip_model"):
        if name.startswith("clip_adapter.prompt_learner"):
            
            if not torch.all(model_weight[name]== model_weight_5000[name]):
                print(name)
        nameList.append(name)
    pass