import torch

def compare_modelweights(w1, w2):
    import pdb; pdb.set_trace()
    m1 = torch.load(w1)
    w_m1 = m1.state_dict()
    w_m2 = torch.load(w2)["model"]
    # w_m2 = m2.state_dict()
    for key, v1 in w_m1.items():
        v2 = w_m2[key]
        import pdb; pdb.set_trace()

if __name__=="__main__":
    w1 = "rerun.pth"
    w2 = "cprompt/output0128end2end/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt__dice08_bce20_layerPre5_learnposition_addcfg_noend2end_lr02_img512_bat32_1/model_final.pth"
    compare_modelweights(w1, w2)
    pass