import torch

def compare_modelweights(w1, w2):
    # import pdb; pdb.set_trace()
    m1 = torch.load(w1)
    w_m1 = m1.state_dict()
    w_m2 = torch.load(w2)["model"]
    # w_m2 = m2.state_dict()
    print("load finish")
    for key, v1 in w_m1.items():
        v2 = w_m2[key]
        sub = torch.sum(v1-v2)
        if sub > 1e-5:
            print(key)
    print("finish")
        # import pdb; pdb.set_trace()
def compare_midvalue(w1, w2):
    m1 = torch.load(w1)
    m2 = torch.load(w2)
    import pdb; pdb.set_trace()

if __name__=="__main__":
    w1 = "rerun.pth"
    w2 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0128end2end/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt__dice08_bce20_layerPre5_learnposition_addcfg_noend2end_lr02_img512_bat32_1/model_final.pth"
    # compare_modelweights(w1, w2)
    w1= "1391_1.pth"
    w2="1391_2.pth"
    compare_midvalue(w1, w2)
    

    pass
