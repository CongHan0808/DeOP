"""
based on detectron2
"""
import os
import glob
import cv2
import functools
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import sys
sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline")
from mask_former import *
from prettytable import PrettyTable
from pycocotools import mask as mask_utils
from PIL import Image

try:
    import wandb
except:
    wandb = None
import json


def mask2seg(
    mask_list, stuff_id_to_contiguous_id, ignore_label=255, include_label=None
):
    masks = [mask_utils.decode(m["segmentation"]) for m in mask_list]
    seg = np.full_like(masks[0], ignore_label)
    for i, mask in enumerate(mask_list):
        if (include_label is not None) and (i not in include_label):
            continue
        seg[masks[i] == 1] = stuff_id_to_contiguous_id[mask["category_id"]]
    return {"file_name": mask_list[0]["file_name"], "seg": seg}

def filter_label(seg,include_label,ignore_label=254,mask=None):
    if mask is not None:
        seg[mask]=ignore_label+1
    for i in np.unique(seg):
        if i not in include_label+[ignore_label]:
            seg[seg==i]=ignore_label
    return seg
def load_jsonfile(file_path, stuff_id_to_contiguous_id, include_label=None):
    with open(file_path) as f:
        pred_list = json.load(f)
    # {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
    # group pred
    print("Loading predictions....")
    preds = {}
    for pred in pred_list:
        if pred["file_name"] not in preds:
            preds[pred["file_name"]] = []
        preds[pred["file_name"]].append(pred)
    preds = [
        mask2seg(v, stuff_id_to_contiguous_id, include_label=include_label)
        for k, v in preds.items()
    ]
    return preds


def main(
    pred_jsonfile,
    gt_dir=None,
    img_dir=None,
    dataset_name="ade20k_sem_seg_val",
    wandb_title: str = None,
    novel_only=False,
):

    metadata = MetadataCatalog.get(dataset_name)
    stuff_id_to_contiguous_id = metadata.stuff_dataset_id_to_contiguous_id
    print(stuff_id_to_contiguous_id)
    class_labels = {i: name for i, name in enumerate(metadata.stuff_classes)}
    include_label = None
    print(stuff_id_to_contiguous_id.values())
    if novel_only:

        novel_id = [
            i
            for i in stuff_id_to_contiguous_id.values()
            if metadata.trainable_flag[i] == 0
        ]
        print(novel_id)
        import pdb; pdb.set_trace()
        novel_class_labels = {k: v for k, v in class_labels.items() if k in novel_id}
        table = PrettyTable(["id", "name"])
        table.add_rows([[k, v] for k, v in novel_class_labels.items()])
        print(table)
        class_labels=novel_class_labels
        class_labels[254]="seen"
    class_labels[255]="ignore"
    if img_dir is None:
        img_dir = metadata.image_root
    if gt_dir is None:
        gt_dir = metadata.sem_seg_root
    print(wandb_title)

    if wandb_title:
        wandb.init(name=wandb_title)
    if "," in pred_jsonfile:
        pred_jsonfile = pred_jsonfile.split(",")
    else:
        pred_jsonfile = [pred_jsonfile]
    pred_jsonfile = [
        [p.split("=")[0], p.split("=")[1]] if "=" in p else ["pred", p]
        for p in pred_jsonfile
    ]
    preds = []
    table = PrettyTable(["File", "Size"])
    for f in pred_jsonfile:
        preds.append(
            [
                f[0],
                load_jsonfile(
                    f[1], stuff_id_to_contiguous_id, include_label=include_label
                ),
            ]
        )
        table.add_row([preds[-1][0], len(preds[-1][1])])
    gt_files = [
        os.path.join(gt_dir, os.path.basename(pred["file_name"]).replace("jpg", "png"))
        for pred in preds[0][1]
    ]
    img_files = [
        os.path.join(img_dir, os.path.basename(pred["file_name"]))
        for pred in preds[0][1]
    ]

    for i, (gfile, img_file) in tqdm(
        enumerate(zip(gt_files, img_files)), total=len(gt_files)
    ):

        gt = cv2.imread(gfile, cv2.IMREAD_GRAYSCALE)
        
        img = np.asarray(Image.open(img_file))

        masks = []
        seen_mask=None
        if novel_only:
            gt=filter_label(gt,include_label=novel_id+[255])
            seen_mask=np.logical_or(gt==254,gt==255)
        gt_labels=np.unique(gt)
        if not any([g in novel_id for g in gt_labels]): 
            print(f"Skip {i} samples as it doesn't contain unseen category.")
            continue
        for pred in preds:
            masks.append(
                {"pred": {"mask_data": pred[1][i]["seg"] if not novel_only else filter_label(pred[1][i]["seg"],include_label=novel_id+[255],mask=seen_mask), "class_labels": class_labels}}
            )
        masks.append({"gt": {"mask_data": gt, "class_labels": class_labels}})
        masks = [
            wandb.Image(img, masks=m, caption=c)
            for m, c in zip(
                masks, [pred_jsonfile[i] for i in range(len(pred_jsonfile))] + ["gt"]
            )
        ]
        wandb.log({"vis": masks})


if __name__ == "__main__":
    import fire

    fire.Fire(main)