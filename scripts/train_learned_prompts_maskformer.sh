export DETECTRON2_DATASETS=datasets
TRAINED_PROMPTS="learnprompt_bs32_10k_4/model_final.pth"
echo ${TRAINED_PROMPTS}

OutPutDir="maskproposalmodel/"
train_net=train_net.py
configfile="configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml"
layermaskvit=[]
python3 ${train_net} --config-file ${configfile} --num-gpus 4 --resume  \
 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} \
 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.LAYERMASKVIT ${layermaskvit} \
 SOLVER.IMS_PER_BATCH 32 MODEL.MASK_FORMER.ONLY_MASKFORMER True
