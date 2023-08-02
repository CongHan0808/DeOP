# Training prompts
export DETECTRON2_DATASETS=datasets
train_py=train_net.py
configfile="configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_bs32_10k.yaml"
outputdir="learnprompt_bs32_10k/model_final.pth"
python3 ${train_py} --config-file ${configfile} --num-gpus 4 OUTPUT_DIR ${outputdir} 

