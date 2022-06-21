# This file contains the class FaceDetector and the individuals we want to use
import copy
import sys

import cv2
import pandas as pd
from facenet_pytorch import MTCNN
import face_detection
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F
from torch.nn.functional import normalize as th_norm

from config import get_parser
from external import model_yolo5
from utils import get_gt_embeddings
from utils import bbox_x1_y1_x2_y2_to_cx_cy_w_h, crop_resize, fixed_image_standardization, add_padding, unpad_bbox


class FaceDetectorBase:
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        self.device = device
        self.feat_extractor = None
        self.det_path = det_path  # In case we use pre-computed detections
        self.results_path = res_path
        # Video info
        self.video_name = video_name
        self.video_fps = video_fps
        self.video_writer = None
        # Politician individuals
        self.model_detection = None
        self.video_dets_df = None
        # Other variables
        self.embs_list = []
        self.faces_frame = None
        # Configuration file arguments
        self.args = None
        self._parse_args_()
        # Initialize face detector and feature extraction individuals
        self._init_detection_model_()
        # Initialise dataframe for pre-computed detections
        if det_path:
            self._init_df_dets_()

    def set_info(self, res_path='data/results', video_name='demo', video_fps=29.97, det_path=None, device=None):
        if device is not None:
            self.device = device

        self.results_path = res_path
        self.video_name = video_name
        self.video_fps = video_fps
        self.det_path = det_path

        if det_path:
            self._init_df_dets_()

    def set_feat_extractor(self, feat_extractor):
        self.feat_extractor = feat_extractor

    def _parse_args_(self):
        parser = get_parser()
        self.args = parser.parse_args()

    def _init_detection_model_(self):
        # Define the individuals detector and feature extraction
        print('Build _init_model_ function for your method')
        pass

    def _init_video_writer_(self, img):
        img_h, img_w = img.shape[0], img.shape[1]
        write_path = (Path(self.results_path) / self.video_name).with_suffix('.mp4')
        self.video_writer = cv2.VideoWriter(str(write_path), cv2.VideoWriter_fourcc(*'MP4V'), self.video_fps, (img_w, img_h))

    def _init_df_dets_(self):
        self.video_dets_df = pd.read_csv(self.det_path)

    def detect(self, img):
        # Input: image in the form of pillow Image
        # Output: bounding boxes and probabilties of the detections
        print('Build detect function for your method')
        return -1, -1

    def crop_and_get_embeddings(self, img, bboxes):
        # Input: image and bounding boxes
        # Output: embeddings per image
        face_embeddings = self.feat_extractor.crop_and_get_embeddings(img, bboxes)
        return face_embeddings

    def get_embeddings(self, face_imgs):
        # Input: batch of cropped face images
        # Output: embeddings per image
        face_embeddings = self.feat_extractor.get_embeddings(face_imgs)
        return face_embeddings

    @staticmethod
    def draw_detections(img, bboxes):
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        for i, bbox in enumerate(bboxes):
            draw.rectangle(bbox.tolist(), width=5)

        img_draw.show()

    @staticmethod
    def draw_detections_cv2(img, bboxes, labels):
        color = (255, 255, 255)
        for bbox, label in zip(bboxes, labels):
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)

        cv2.imshow('Faces', img)
        cv2.waitKey(1)

    def write_in_video(self, img, bboxes, labels, dists, frame_id=-1):
        if self.video_writer is None:
            self._init_video_writer_(img)

        color = (255, 255, 255)
        for bbox, label, dist in zip(bboxes, labels, dists):
            if sum(bbox) == 0:  # bbox = [0, 0, 0, 0]
                continue

            if self.args.gender:
                dist_str = str(round(dist * 100, 1))
                if label == 'female':
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                elif label == 'male':
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                else:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            else:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(img, label, (int(bbox[0])+5, int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            if frame_id > -1:
                cv2.putText(img, str(frame_id), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        self.video_writer.write(img)

    def concat_embs_to_extract(self, res_file, frame_ID, bboxes, probs, embs):
        for bbox, prob_det, emb in zip(bboxes, probs, embs.cpu().numpy()):
            bbox_c_w = bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox)
            res_row_list = [
                res_file.stem,
                '-1',
                str(int(frame_ID)),
                str(int(bbox_c_w[0])),
                str(int(bbox_c_w[1])),
                str(int(bbox_c_w[2])),
                str(int(bbox_c_w[3])),
                str(prob_det)[:4],  # 2 decimals
                emb
            ]

            self.embs_list.append(res_row_list)

    def detect_and_write_to_file(self, frame, frame_ID, w_res_file, res_file, **kwargs):
        # kwargs variables (flags)
        flag_only_detections = kwargs.get('flag_only_detections', False)
        flag_extract_embs = kwargs.get('flag_extract_embs', False)
        flag_save_vid = kwargs.get('flag_save_vid', False)

        # Input: frame and file writer (already opened)
        if type(frame) != np.ndarray:
            # Video has errors or something
            return

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.det_path is None:
            bboxes, probs = self.detect(img)
        else:
            df = self.video_dets_df[self.video_dets_df['frame'] == frame_ID]
            x1s = (df['cx'] - df['w']/2).to_numpy()
            y1s = (df['cy'] - df['h']/2).to_numpy()
            x2s = (df['cx'] + df['w']/2).to_numpy()
            y2s = (df['cy'] + df['h']/2).to_numpy()
            probs = df['prob_det'].to_numpy()
            bboxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)]
            probs = [prob for prob in probs]

        # For now, no labels
        if bboxes is not None:
            if flag_only_detections:
                # Fill elements of the csv file
                labels = ['0' for _ in bboxes]
                emb_dists = [-1 for _ in bboxes]
            else:
                # Get face embeddings
                embs_face = self.crop_and_get_embeddings(img, bboxes)
                if flag_extract_embs and embs_face is not None:
                    self.concat_embs_to_extract(res_file, frame_ID, bboxes, probs, embs_face)
                    return

                labels, emb_dists = self.feat_extractor.assign_label(embs_face)
                # labels = [str(i) for i in range(len(bboxes))]
        else:
            bboxes = [[0, 0, 0, 0]]
            probs = [0]
            labels = ['0']
            emb_dists = [-1]

        # res_row = ','.join(['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID']) + '\n'
        flag_written = False
        for bbox, label, prob_det, emb_dist in zip(bboxes, labels, probs, emb_dists):
            if label == '0' and not flag_only_detections:  # Ignore other detections
                continue

            bbox_c_w = bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox)
            res_row_list = [
                res_file.stem,
                label,
                str(frame_ID),
                str(int(bbox_c_w[0])),
                str(int(bbox_c_w[1])),
                str(int(bbox_c_w[2])),
                str(int(bbox_c_w[3])),
                str(prob_det)[:4],  # 2 decimals
                str(emb_dist)[:4]
            ]
            res_row = ','.join(res_row_list) + '\n'
            w_res_file.write(res_row)
            flag_written = True

        # To have the information on which frames are visited, if there is no individual detected, write a line indicating the frame
        if not flag_written:
            res_row_list = [
                res_file.stem,
                '0',
                str(frame_ID),
                str(0),
                str(0),
                str(0),
                str(0),
                str(0),
                '-1'
            ]
            res_row = ','.join(res_row_list) + '\n'
            w_res_file.write(res_row)

        if flag_save_vid:
            self.write_in_video(frame, bboxes, labels, dists=emb_dists, frame_id=frame_ID)


class FeatureExtractor:
    def __init__(self, individuals_path, feat_extractor='resnetv1', feat_input_size=112, device='cpu', **kwargs):
        # Input arguments
        self.feat_extractor_name = feat_extractor
        self.feat_input_size = feat_input_size
        self.device = device
        # Other arguments
        # These arguments are for online computation
        self.mod_feat = kwargs.get('mod_feat', 'knn_1')
        self.dist = kwargs.get('dist', 0.3)
        # Variables
        self.model = None
        self.individuals_dict = None
        # Initialize variables
        # Dictionary for individuals
        self._init_individuals_dict_()
        # Feature extractor to use
        self._init_feats_model_()
        # For backwards compatibility for extracting face embeddings of individuals we need a face detector
        if kwargs.get('face_detector', False):
            self.face_detector = kwargs['face_detector']
        # GT embeddings
        self._init_gt_embeddings_(individuals_path)

    def _init_individuals_dict_(self):
        self.individuals_dict = dict()

    def _init_feats_model_(self):
        # Feature extractor model
        if self.feat_extractor_name == 'resnetv1':
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        elif self.feat_extractor_name == 'resnetv1-casio':
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='casia-webface', device=self.device).eval()
        elif self.feat_extractor_name == 'magface':
            from collections import namedtuple
            from external.model_magface import builder_inf
            args_magface = {
                'arch': 'iresnet100',
                'embedding_size': 512,
                'cpu_mode': True if self.device == 'cpu' else False,
                'device': self.device,
                'resume': 'src/external/magface_epoch_00025.pth'  # Make sure to have this inside "external"
            }
            # Dictionary to object
            # https://www.kite.com/python/answers/how-to-convert-a-dictionary-into-an-object-in-python
            args_magface_object = namedtuple("MagFaceArgs", args_magface.keys())(*args_magface.values())
            self.model = builder_inf(args_magface_object)
            self.model.to(self.device).eval()

    def _init_gt_embeddings_(self, individuals_path):
        gt_imgs, gt_embs, gt_labs = get_gt_embeddings(self.face_detector, self, individuals_path)
        for img, embedding, label in zip(gt_imgs, gt_embs, gt_labs):
            if label not in self.individuals_dict.keys():
                self.individuals_dict[label] = dict()
                self.individuals_dict[label]['emb'] = []

            self.individuals_dict[label]['emb'].append(embedding)

        # For unique values
        for label in set(gt_labs):
            self.individuals_dict[label]['emb'] = torch.stack(self.individuals_dict[label]['emb'])
            if self.mod_feat == 'mean_dist':
                self.individuals_dict[label]['emb'] = torch.mean(self.individuals_dict[label]['emb'], dim=0).unsqueeze(0)
            elif self.mod_feat == 'min_plus_mean_dist' or self.mod_feat == 'all_plus_mean_dist' or self.mod_feat == 'min_plus_mean_nowhite_dist':
                mean_emb = torch.mean(self.individuals_dict[label]['emb'], dim=0).unsqueeze(0)
                self.individuals_dict[label]['emb'] = torch.cat((self.individuals_dict[label]['emb'], mean_emb))

    def crop_images(self, img, bboxes, output_size=160, tolerance=0):
        # Crop face images and put them into a torch tensor (not normalized)
        bboxes_int = [[int(bbox[0] - (tolerance / 2)), int(bbox[1] - (tolerance / 2)),
                       int(bbox[2] + (tolerance / 2)), int(bbox[3] + (tolerance / 2))] for bbox in bboxes]
        face_imgs = [crop_resize(img, bbox, output_size) for bbox in bboxes_int]
        return face_imgs

    def crop_and_batch(self, img, bboxes, output_size=160, tolerance=0, opt_norm=False):
        # Crop face images and put them into a torch tensor (not normalized)
        face_imgs = self.crop_images(img, bboxes, output_size, tolerance)
        if opt_norm:
            face_imgs_th = torch.stack([F.to_tensor(face_img) for face_img in face_imgs])
        else:
            face_imgs_th = torch.stack([F.to_tensor(np.float32(face_img)) for face_img in face_imgs])

        return face_imgs_th

    def assign_label(self, det_embeddings):
        # The input are the embeddings from detections. Assign a label (or not) to a detection
        # I want to check ASAP the results, but do this with a distance matrix in the future, and that second loop is horrible
        labels = ['0'] * det_embeddings.shape[0]  # List creation in Python, lol
        emb_dist_list = [100] * det_embeddings.shape[0]
        for idx, emb in enumerate(det_embeddings):
            min_dist = 100
            res_emb_dist = 100
            k_pol_adapt = None
            for k_pol in self.individuals_dict.keys():
                # Computing the distance (equivalent to 1-similarity)
                # If cosine similarity = 0, vectors are orthogonal
                # https://stackoverflow.com/questions/58381092/difference-between-cosine-similarity-and-cosine-distance
                emb_dist = 1 - torch.cosine_similarity(self.individuals_dict[k_pol]['emb'], emb.unsqueeze(0))
                # Here put the combination of features using different approaches
                if self.mod_feat.find('knn') > -1:
                    # e.g. knn_1_adapt
                    # Get minimum between the specified K and the amount of embeddings per individual
                    k_opt = self.mod_feat.split('_')[1]
                    if k_opt == 'max':
                        k_neighs = len(self.individuals_dict[k_pol]['emb'])
                    else:
                        k_neighs = min(int(k_opt), len(self.individuals_dict[k_pol]['emb']))
                    num_close_elems = sum(emb_dist < self.dist)  # Check all the distances lower than the threshold
                    if num_close_elems >= k_neighs:
                        res_emb_dist = torch.mean(
                            emb_dist)  # 2 neighbors with 0.2 and 0.2 should be closer to 0.15 0.3 (?)
                    else:
                        res_emb_dist = 100

                if res_emb_dist < min_dist and res_emb_dist < 0.3:
                    min_dist = res_emb_dist
                    labels[idx] = k_pol
                    emb_dist_list[idx] = res_emb_dist.item()
                    if self.mod_feat.find('adapt') > -1 and min_dist < 0.15:
                        k_pol_adapt = copy.copy(k_pol)

            # Add embedding to politician model
            if k_pol_adapt is not None:
                self.individuals_dict[k_pol_adapt]['emb'] = torch.cat(
                    [self.individuals_dict[k_pol_adapt]['emb'], emb.unsqueeze(0)])

        return labels, emb_dist_list

    def crop_and_get_embeddings(self, img, bboxes):
        if self.feat_extractor_name == 'magface':
            # Magface has BGR inputs
            img_in = np.array(img)
            img_in = img_in[:, :, ::-1]
            img_in = Image.fromarray(img_in)
        else:
            img_in = img
        # Get cropped and "normalized" image tensor
        opt_norm = True if self.feat_extractor_name == 'magface' else False
        face_imgs_th = self.crop_and_batch(img_in, bboxes, output_size=self.feat_input_size, opt_norm=opt_norm)

        if self.feat_extractor_name == 'magface':
            face_imgs = F.normalize(face_imgs_th, mean=[0., 0., 0.], std=[1., 1., 1.])
        else:
            face_imgs = fixed_image_standardization(face_imgs_th)

        img_embedding = self.get_embeddings(face_imgs)
        return img_embedding

    @torch.inference_mode()
    def get_embeddings(self, face_imgs):
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = self.model(face_imgs.to(self.device))
        if self.feat_extractor_name == 'magface':
            # Normalize feature vector w.r.t the magnitude
            img_embedding = th_norm(img_embedding)

        return img_embedding


class FaceDetectorMTCNN(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self, input_size=160):
        # Detection model
        self.model_detection = MTCNN(image_size=input_size, margin=0, min_face_size=20, keep_all=True, device=self.device).eval()  # keep_all --> returns all bboxes in video

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        boxes, probs = self.model_detection.detect(img)
        return boxes, probs


class FaceDetectorDFSD(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self):
        # Detection model
        self.model_detection = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device=self.device)

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        boxes_probs = self.model_detection.detect(open_cv_image)
        boxes = boxes_probs[:, :4]
        probs = boxes_probs[:, -1]
        if len(boxes) == 0:
            boxes = None
            probs = None
        return boxes, probs


class FaceDetectorYOLO5(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self):
        # Detection model
        self.model_detection = model_yolo5.YOLO5FACE(conf_thresh=.5, nms_iou_thresh=.3, device=self.device)

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        open_cv_image = np.array(img)
        boxes_probs, _ = self.model_detection.detect(open_cv_image)

        if len(boxes_probs) == 0:
            boxes = None
            probs = None
        else:
            boxes = boxes_probs[:, :4]
            probs = boxes_probs[:, -1]

        return boxes, probs
