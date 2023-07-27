export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_2/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_3/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_4/model_final.pth"
echo ${TRAINED_PROMPTS}

#OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-1"
# #python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 4 --eval-only --resume  MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} OUTPUT_DIR ${OutPutDir}
# OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-2"
# python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 4 --eval-only --resume  MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} OUTPUT_DIR ${OutPutDir}
# OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-3"
# python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 4 --eval-only --resume  MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} OUTPUT_DIR ${OutPutDir}

OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify"
proposalmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify/model_final.pth"
proposalmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify/model_final_noclip.pth"
OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0128end2end/R101c_decoder_1layer_vit16_featup_delnorm@decoder_query2D_maskvit_noatten_learnprompt__dice08_bce20_layerPre5_learnposition_addcfg_noend2end_lr02_img512_bat32_1"
#CUDA_VISIBLE_DEVICES=1 python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask.yaml --num-gpus 1 --eval-only --resume MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT -1.0 SOLVER.TEST_IMS_PER_BATCH 4 OUTPUT_DIR ${OutPutDir}
promptlearn="predefined"
promptlearn="learnable"
promptTemplate=["a sculpture of a {}."]
clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/RN50.pt"
clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/ViT-B-16.pt"
#configfile="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img384.yaml"
configfile="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img512.yaml"
maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D"
maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit"
python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file ${configfile} --num-gpus 4 --resume --eval-only \
 MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT -1.0 SOLVER.TEST_IMS_PER_BATCH 1 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn} SOLVER.IMS_PER_BATCH 32 \
 SOLVER.BASE_LR 0.0002 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME ${clipmodel} MODEL.WEIGHTS ${proposalmodel} MODEL.META_ARCHITECTURE ${maskformermodel} MODEL.NUM_DECODER_LAYER 1  ORACLE False INPUT.MIN_SIZE_TEST 512 TEST.EVAL_PERIOD 5000 INPUT.SIZE_DIVISIBILITY 512 MODEL.MASK_FORMER.DECODER_DICE_WEIGHT 0.8 MODEL.MASK_FORMER.DECODER_CE_WEIGHT 2.0 MODEL.CLIP_ADAPTER.LEARN_POSITION True MODEL.CLIP_ADAPTER.POSITION_LAYERS '[1,2,3,4,5]' MODEL.CLIP_ADAPTER.END2ENDTRAIN True MODEL.CLIP_ADAPTER.PS_SHORTCUT 1.0
# [1,2,3,4,5,6,7,8,9,10,11]
#random seed 46018321
#python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask.yaml --num-gpus 1 --eval-only --resume MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT -1.0 SOLVER.TEST_IMS_PER_BATCH 4 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn} MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME ${clipmodel}
