export DETECTRON2_DATASETS=datasets
TRAINED_PROMPTS=promptproposal/learnprompt_bs32_10k/model_final.pth"

OutPutDir="deop/train_log"
proposalmodel=${MaskProposalModel}
promptlearn="learnable"
clipmodel="clip/ViT-B-16.pt"
configfile="configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img512.yaml"

maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit"
train_net=train_net.py

python3 ${train_net} --config-file ${configfile} --num-gpus 4 --resume \
 MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT -1.0 SOLVER.TEST_IMS_PER_BATCH 1 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn} SOLVER.IMS_PER_BATCH 16 \
 SOLVER.BASE_LR 0.0002 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME ${clipmodel} MODEL.WEIGHTS ${proposalmodel} MODEL.META_ARCHITECTURE ${maskformermodel} MODEL.NUM_DECODER_LAYER 1 \
 ORACLE False INPUT.MIN_SIZE_TEST 512 TEST.EVAL_PERIOD 5000 INPUT.SIZE_DIVISIBILITY 512 MODEL.MASK_FORMER.DECODER_DICE_WEIGHT 0.8 MODEL.MASK_FORMER.DECODER_CE_WEIGHT 2.0 MODEL.CLIP_ADAPTER.LEARN_POSITION True \
 MODEL.CLIP_ADAPTER.POSITION_LAYERS '[1,2,3,4,5]' MODEL.CLIP_ADAPTER.END2ENDTRAIN True MODEL.CLIP_ADAPTER.PS_SHORTCUT 1.0
