export DETECTRON2_DATASETS=datasets
proposalmodel="model_final.pth"
OutPutDir="verifycoco"

promptlearn="learnable"

clipmodel="clip/ViT-B-16.pt"

configfile="configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img512.yaml"
maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit"
train_net=train_net.py
python3 ${train_net} --config-file ${configfile} --num-gpus 4 --eval-only \
 SOLVER.TEST_IMS_PER_BATCH 1 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn}  \
 MODEL.WEIGHTS ${proposalmodel} MODEL.META_ARCHITECTURE ${maskformermodel} \
 MODEL.NUM_DECODER_LAYER 1  ORACLE False INPUT.MIN_SIZE_TEST 512 INPUT.SIZE_DIVISIBILITY 512 \
 MODEL.CLIP_ADAPTER.LEARN_POSITION True MODEL.CLIP_ADAPTER.POSITION_LAYERS '[1,2,3,4,5]' 
