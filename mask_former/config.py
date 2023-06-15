# -*- codpromptTemplateing: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_mask_former_default_config(cfg):
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # 2023-01-17更新 处理maskformer和clip输入尺寸不一致
    cfg.INPUT.SIZE_CLIP = -1
    # 2023-01-17更新 处理learnpositon 尺寸
    cfg.INPUT.LEARNPOSITIONRES = 384
    # 2023-01-31更新 src_output 通过sigmoid超参
    cfg.INPUT.OUTPUT_SIG = 3.0
    # 2023-04-27更新 增加caption信息
    cfg.INPUT.CAPTION_FILE_TRAIN = ""
    cfg.INPUT.CAPTION_FILE_TEST = ""
    cfg.INPUT.CAPTION_TOKENIZE_TRUNCATE= False
    cfg.INPUT.CAPTION_TOKENIZE_CONTEXT_LENGTH = 77

    
    

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.DECODER_DICE_WEIGHT =0.0
    cfg.MODEL.MASK_FORMER.DECODER_CE_WEIGHT = 1.0

    cfg.MODEL.MASK_FORMER.MASKDOWNSAMPLE = False
    
    # 2023-02-06 添加训练时num_class应对不同数据集
    cfg.MODEL.MASK_FORMER.LOSS_NUM_CLASS = 156 # 默认值为coco-stuff seen类别数量
    # kd loss相关
    cfg.MODEL.MASK_FORMER.CLIP_KD_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.CLIP_KD_LOSS = False
    cfg.MODEL.MASK_FORMER.CLIP_KD_PROJ = False

    # SAM 相关
    cfg.MODEL.MASK_FORMER.SAM_CKPT = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/segment-anything/segment_anything/pretrain_weights/sam_vit_b_01ec64.pth"
    cfg.MODEL.MASK_FORMER.SAM_MODELTYPE = "vit_b"
    cfg.MODEL.MASK_FORMER.SAM_POINTS_PER_SIDE = 16



    # 评估mask proposal质量：
    cfg.MODEL.EVALUATIONTYPE = CN()
    cfg.MODEL.EVALUATIONTYPE.SEG_AP = False
    cfg.MODEL.EVALUATIONTYPE.SEG_AP_OUTPUT = False

    # backbone修改为clip image encoder
    cfg.MODEL.BACKBONE_CLIP = False

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"


    ############################################
    # decoder layer 
    cfg.MODEL.NUM_DECODER_LAYER = 5
    cfg.MODEL.DECODER_MASK_FEATURE = False
    

    cfg.MODEL.DECODER_CONV_LAYERS = 1
    cfg.MODEL.DECODER_CONV_ACTIVATION = "sigmoid"

    cfg.MODEL.CRITERION_LABELLOSS = False
    ############################################


    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


def add_our_config(cfg):
    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_WITH_PRIOR = True
    cfg.PSEUDO_REJECT_THRESHOLD = 0.0
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.SLIDING_TILE_SIZE = 224
    cfg.TEST.SLIDING_OVERLAP = 2 / 3.0
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.DATASETS.SAMPLE_SEED = 0
    # whether to use dense crf
    cfg.TEST.DENSE_CRF = False
    # embedding head
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2
    # clip_adapter
    cfg.MODEL.CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER = "imagenet"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = ["a sculpture of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_ADAPTER.MASK_THR = 0.5
    cfg.MODEL.CLIP_ADAPTER.MASK_MATTING = False
    cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT = 0.8
    # 2023-01-17 更新，增加end2end训练判断flag
    cfg.MODEL.CLIP_ADAPTER.END2ENDTRAIN = False
    cfg.MODEL.CLIP_ADAPTER.LEARN_POSITION = False
    cfg.MODEL.CLIP_ADAPTER.POSITION_LAYERS = []

    # 2023-02-09 更新，增加vpt
    cfg.MODEL.CLIP_ADAPTER.LEARN_TOKEN = False
    cfg.MODEL.CLIP_ADAPTER.VPT_NUM_TOKEN = 5

    cfg.MODEL.CLIP_ADAPTER.LAYERMASKVIT = [11,]
    cfg.MODEL.CLIP_ADAPTER.MASKSELFATTN=False
    cfg.MODEL.CLIP_ADAPTER.MASKSELFATTNSOFTMAX = False
    cfg.MODEL.CLIP_ADAPTER.PS_SHORTCUT = 1.0
    
    #
    cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER = False
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_LEARNER = "predefined"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = [
        "a sculpture of a {}."
    ]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_CHECKPOINT = ""

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "zero_shot_seg"
    cfg.WANDB.NAME = None
    cfg.WANDB.ENTITY = ""


def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    add_mask_former_default_config(cfg)
    add_our_config(cfg)
