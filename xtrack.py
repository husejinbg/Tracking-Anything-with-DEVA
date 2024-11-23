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


class XTrack:
    def __init__(self, args, cfg: dict, output_root: str, video_name: str, mask_alpha: float = 0.5,
                 annotate_boxes: bool = True, annotate_labels: bool = True, show_only_annotations: bool = False,
                 save_masks_seperately: bool = False, fps: int = 30, output_video_path: str = "output.mp4"):
        
        self.cfg = cfg
        self.fps = fps
        self.ti = 0
        self.output_video_path = output_video_path
        
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
