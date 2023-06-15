# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
from unicodedata import category
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator


class GeneralizedSemSegEvaluatorClassAgnostic(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.countIoU = 0
        self.countAllMask = 0

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            # output = self.post_process_func(
            #     output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            # )
            # output = output.argmax(dim=0).to(self._cpu_device)
            # # import pdb; pdb.set_trace()
            # pred = np.array(output, dtype=np.int)
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            # self._conf_matrix += np.bincount(
            #     (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            #     minlength=self._conf_matrix.size,
            # ).reshape(self._conf_matrix.shape)
            # import pdb; pdb.set_trace()
            mask_pred_results = output["pred_masks"].sigmoid()
            image = np.array(Image.open(input["file_name"]))
            # masks_output = self.post_process_func(
            #     output["pred_masks"], image=np.array(Image.open(input["file_name"]))
            # )
            
            masks_output = F.interpolate(
                mask_pred_results.unsqueeze(0),
                size=(image.shape[0], image.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            # import pdb; pdb.set_trace()
            for idx, mask_output in enumerate(masks_output.squeeze(0)):
                gt_idall = np.unique(gt)
                mask_out = np.zeros_like(gt)
                # import pdb; pdb.set_trace()
                mask_out[mask_output.to("cpu") > 0.5] = 1.0
                if(np.sum(mask_out)> 0):
                    self.countAllMask += 1
                else:
                    continue
                for gt_idx in gt_idall:
                    mask_gt = np.zeros_like(gt)
                    mask_gt[gt == gt_idx] = 1.0
                    mask_uion = mask_out + mask_gt
                    tp = np.sum(mask_uion > 1.5)
                    uion = np.sum(mask_uion > 0.8)
                    if(uion > 0):
                        iou = tp / uion
                    if iou > 0.5:
                        self.countIoU += 1
                        break

            print(f"countIoU: {self.countIoU}, countAllMask: {self.countAllMask}")
            # self._predictions.extend(self.encode_json_sem_seg(, input["file_name"]))
            
            '''
            encodeJsonSemSeg = self.encode_json_sem_seg(pred, input["file_name"])
            
            print(len(encodeJsonSemSeg))
            categoryIds = ""
            # Image.fromarray(pred.astype(np.uint8)).save("../output/tmp_img/out_"+encodeJsonSemSeg[-1]["file_name"].split("/")[-1].replace("jpg", "png"), "PNG")
            for label in np.unique(pred):
                if self._contiguous_id_to_dataset_id is not None:
                    assert (
                        label in self._contiguous_id_to_dataset_id
                    ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                    dataset_id = self._contiguous_id_to_dataset_id[label]
                else:
                    print("no contiguous_id_to_dataset_id")
                    dataset_id = int(label)
                categoryIds += str(dataset_id) + " "
            print(encodeJsonSemSeg[-1]["file_name"], categoryIds)
        print(self._contiguous_id_to_dataset_id)
        '''

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        # import pdb; pdb.set_trace()
        if self._distributed:
            # synchronize()
            # conf_matrix_list = all_gather(self._conf_matrix)
            # self._predictions = all_gather(self._predictions)
            # self._predictions = list(itertools.chain(*self._predictions))
            countIoU_list = all_gather(self.countIoU)
            countAllMask_list = all_gather(self.countAllMask)
            if not is_main_process():
                return

            # self._conf_matrix = np.zeros_like(self._conf_matrix)
            # for conf_matrix in conf_matrix_list:
            #     self._conf_matrix += conf_matrix
            self.countIoU = 0
            self.countAllMask = 0
            for countIoU, countAllMask in zip(countIoU_list, countAllMask_list):
                self.countIoU += countIoU
                self.countAllMask += countAllMask

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))
        res={}
        if(self.countAllMask > 0):
            res["AP"] = self.countIoU / self.countAllMask
        else:
            res["AP"] = 0

        # acc = np.full(self._num_classes, np.nan, dtype=np.float)
        # iou = np.full(self._num_classes, np.nan, dtype=np.float)
        # tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        # pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        # class_weights = pos_gt / np.sum(pos_gt)
        # pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        # acc_valid = pos_gt > 0
        # acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        # iou_valid = (pos_gt + pos_pred) > 0
        # union = pos_gt + pos_pred - tp
        # iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        # macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        # miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        # fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        # pacc = np.sum(tp) / np.sum(pos_gt)

        # res = {}
        # res["mIoU"] = 100 * miou
        # res["fwIoU"] = 100 * fiou
        # for i, name in enumerate(self._class_names):
        #     res["IoU-{}".format(name)] = 100 * iou[i]
        # res["mACC"] = 100 * macc
        # res["pACC"] = 100 * pacc
        # for i, name in enumerate(self._class_names):
        #     res["ACC-{}".format(name)] = 100 * acc[i]
        # if self._evaluation_set is not None:
        #     for set_name, set_inds in self._evaluation_set.items():
        #         iou_list = []
        #         set_inds = np.array(set_inds, np.int)
        #         mask = np.zeros((len(iou),)).astype(np.bool)
        #         mask[set_inds] = 1
        #         miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
        #         pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
        #         res["mIoU-{}".format(set_name)] = 100 * miou
        #         res["pAcc-{}".format(set_name)] = 100 * pacc
        #         iou_list.append(miou)
        #         miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
        #         pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
        #         res["mIoU-un{}".format(set_name)] = 100 * miou
        #         res["pAcc-un{}".format(set_name)] = 100 * pacc
        #         iou_list.append(miou)
        #         res["hIoU-{}".format(set_name)] = (
        #             100 * len(iou_list) / sum([1 / iou for iou in iou_list])
        #         )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg_ap": res})
        self._logger.info(results)
        return results
