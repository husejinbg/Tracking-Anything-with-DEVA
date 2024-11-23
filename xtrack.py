import os
from os import path

import torch
import numpy as np

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import flush_buffer
from deva.ext.grounding_dino import get_grounding_dino_model

from deva.ext.with_text_processor import process_frame_with_text as process_frame_text
import cv2

import json


class XTrack:
    def __init__(self, args, cfg: dict, output_root: str, video_name: str, mask_alpha: float = 0.5,
                 annotate_boxes: bool = True, annotate_labels: bool = True, show_only_annotations: bool = False,
                 save_masks_seperately: bool = False, fps: int = 30, output_video_path: str = "output.mp4"):
        
        self.cfg = cfg
        self.fps = fps
        self.ti = 0
        self.output_video_path = output_video_path
        self.output_root = output_root
        self.video_name = video_name
        self.visualize_postfix = 'Visualizations'
        self.output_postfix = 'Annotations'
        
        torch.autograd.set_grad_enabled(False)

        # for id2rgb
        np.random.seed(42)

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        # Load our checkpoint
        deva_model = DEVA(cfg).cuda().eval()
        if args.model is not None:
            model_weights = torch.load(args.model)
            deva_model.load_weights(model_weights)
        else:
            raise ValueError("No model provided")

        self.gd_model, self.sam_model = get_grounding_dino_model(cfg, 'cuda')

        self.deva = DEVAInferenceCore(deva_model, config=cfg)
        self.deva.next_voting_frame = cfg['num_voting_frames'] - 1
        self.deva.enabled_long_id()

        self.result_saver = ResultSaver(output_root, video_name, dataset='demo', object_manager=self.deva.object_manager,
                            mask_alpha=mask_alpha, annotate_boxes=annotate_boxes, annotate_labels=annotate_labels,
                            show_only_annotations=show_only_annotations, save_masks_seperately=save_masks_seperately)
        self.writer_initialized = False

    def process_single_frame(self, frame: np.ndarray, frame_name: str):
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            if not self.writer_initialized:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'vp80'), self.fps, (w, h))
                self.writer_initialized = True
                self.result_saver.writer = self.writer

            process_frame_text(self.deva,
                                self.gd_model,
                                self.sam_model,
                                frame_name,
                                self.result_saver,
                                self.ti,
                                image_np=frame)
            
            self.ti += 1

        

    def finish(self):
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            flush_buffer(self.deva, self.result_saver)
        self.writer.release()
        self.deva.clear_buffer()

    def _retrieve_results(self, frame_name: str):
        output_path = path.join(self.output_root, self.output_postfix, self.video_name)
        visualizations_path = path.join(self.output_root, self.visualize_postfix, self.video_name)
        frame_name_no_ext = frame_name.split('.')[0]
        with open(path.join(output_path, f"{frame_name_no_ext}_info.json"), 'r') as f:
            info = json.load(f)

        mask_data = []

        for key, value in info.items():
            mask_name = key
            label = " ".join(value.split()[0:-1])
            confidence = float(value.split()[-1])
            mask_img = cv2.imread(path.join(output_path, mask_name))
            mask_data.append((mask_img, label, confidence))

        return mask_data



