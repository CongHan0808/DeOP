export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
proposalmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0128end2end/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt__dice08_bce20_layerPre5_learnposition_addcfg_noend2end_lr02_img512_bat32_1/model_final.pth"
OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/DeOP/prompt/output0217end2end_ablation/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt__dice08_bce20_layerPre5_learnposition_addcfg_noend2end_lr02_img512_bat32_1_verifycoco3/"

promptlearn="learnable"

clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/ViT-B-16.pt"

configfile="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/DeOP/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img512.yaml"
maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit"
train_net=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/DeOP/train_net.py
python3 ${train_net} --config-file ${configfile} --num-gpus 4 --eval-only \
 SOLVER.TEST_IMS_PER_BATCH 1 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn}  \
 MODEL.WEIGHTS ${proposalmodel} MODEL.META_ARCHITECTURE ${maskformermodel} \
 MODEL.NUM_DECODER_LAYER 1  ORACLE False INPUT.MIN_SIZE_TEST 512 INPUT.SIZE_DIVISIBILITY 512 \
 MODEL.CLIP_ADAPTER.LEARN_POSITION True MODEL.CLIP_ADAPTER.POSITION_LAYERS '[1,2,3,4,5]' 
