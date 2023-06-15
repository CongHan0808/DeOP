import json
from pathlib import Path


caption_train2017 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets/coco/annotations/captions_train2017.json"
caption_val2017 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets/coco/annotations/captions_val2017.json"
captions_file = [caption_train2017, caption_val2017]
captions_file_new = ["captions_train2017_new.json", "captions_val2017_new.json"]
cap_path = Path(caption_train2017).parent
for idx, caption_file in enumerate(captions_file):
    annos_new = {}
    with open(caption_file,"r") as f:
        
        infos=json.load(f)
        images = infos["images"]
        annotations = infos["annotations"]
        
        for annotation in annotations:
            image_id = annotation["image_id"]
            imgfile = str(image_id).zfill(12) + ".jpg"
            caption = annotation["caption"]
            if imgfile in annos_new:
                caps = annos_new[imgfile]
                caps.append(caption)
            else:
                caps = []
                caps.append(caption)
                annos_new[imgfile]=caps
    with open(cap_path / captions_file_new[idx], "w") as f:
        result = json.dump(annos_new, f)

    

