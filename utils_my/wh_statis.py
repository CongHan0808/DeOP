import cv2
from pathlib import Path
import numpy as np
def get_dataset_wh(prefix,pathDataset, wh_txt):
    pathImgs=Path(prefix) / pathDataset
    imgList = pathImgs.glob("*.jpg")
    f = open(wh_txt, "w")
    for imgFile in imgList:
        img = cv2.imread(str(imgFile))
        h,w,d = img.shape
        info=str(imgFile) + " " + str(img.shape[0])+" " + str(img.shape[1]) + "\n"
        f.write(info)
        # import pdb; pdb.set_trace()
    f.close()
    
    pass
def analyse_wh(wh_txt):
    with open(wh_txt) as f:
        infos = f.readlines()
        imgSize = []
        for info in infos:
            infoList = info.rstrip("\n").split(" ")
            h = int(infoList[1])
            w = int(infoList[2])
            size = [h,w]
            imgSize.append(size)
        # import pdb; pdb.set_trace()
        imgSizeNp = np.array(imgSize)
        imgSizeMax=imgSizeNp.max(axis = 1)
        imgSizeMin = imgSizeNp.min(axis=1)
        sizeMin =[]
        countPre = 0
        for idx in range(300, 660, 50):
            count = (imgSizeMin <= idx).sum()
            if len(sizeMin) > 0:
                # import pdb ;pdb.set_trace()
                countPre += sizeMin[-1]
                count = count - countPre
            sizeMin.append(count)
        print(sizeMin)
        
        
    pass

if __name__=="__main__":
    prefix = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets"
    pathDataset = "coco/train2017"
    wh_txt = "wh_datataset_train2017.txt"
    # get_dataset_wh(prefix, pathDataset, wh_txt)
    analyse_wh(wh_txt)

    pass