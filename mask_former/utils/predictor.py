# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class DeOPPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                # original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
           
            image = torch.as_tensor(original_image.transpose(2, 0, 1).copy())
            # image = original_image.transpose(2, 0, 1)
           
            inputs = {"image": image, "height": height, "width": width, "class_names": class_names, \
                      "meta":{"dataset_name":"coco_2017_test_stuff_sem_seg"}}
            predictions = self.model([inputs], class_names)[0]
            return predictions

class DeOPVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output
class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        # import pdb;pdb.set_trace()
        self.class_names = [ c.strip() for c in self.metadata.stuff_classes]
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = DeOPPredictor(cfg)

    def run_on_image(self, image, class_names):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if not class_names or len(class_names) == 0:
            visualizer = DeOPVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=self.class_names)
        else:
            visualizer = DeOPVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        
        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            sam_masks = predictions["pred_sam_masks"]
            # blank_area = (r[0] == 0)
            # import pdb; pdb.set_trace()
            # classOneArea = (r[0] == 1)
            # import pdb; pdb.set_trace() 
            pred_mask = r.argmax(dim=0).to('cpu')
            blank_area = (sam_masks.sum(dim = 0) == 0)
            pred_mask[blank_area] = 255
            
            pred_mask = np.array(pred_mask, dtype=np.int)
            # import pdb;pdb.set_trace()
            vis_output = visualizer.draw_sem_seg(
                pred_mask
            )
        else:
            raise NotImplementedError

        return predictions, vis_output