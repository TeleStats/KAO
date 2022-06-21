# Adapted YOLO5-face model from https://github.com/deepcam-cn/yolov5-face

import cv2
import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np
import torch

from src.external.yolo5_face.utils_yolo.datasets import letterbox
from src.external.yolo5_face.utils_yolo.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh


class YOLO5FACE:
    def __init__(self, conf_thresh=.5, nms_iou_thresh=.3, device='cpu'):
        # Variables
        self.model = None
        self.device = device
        self.conf_thresh = conf_thresh
        self.nms_iou_thresh = nms_iou_thresh
        # Init variables
        self.__build_model__()

    def __build_model__(self):
        # Problem when loading the YOLO model (pickle requires a module named "models" to find the different modules)
        sys.path.append(os.path.join(os.getcwd(), 'src/external/yolo5_face'))
        weights_file = f'{os.getcwd()}/src/external/yolo5_face/models/yolov5m-face.pt'
        self.model =  torch.load(weights_file, map_location=self.device)['model'].float().fuse().eval()  # load FP32 model

    def preprocess_img(self, img):
        h0, w0 = img.shape[:2]  # orig hw
        img_size = max(h0, w0)

        imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        img_letter = letterbox(img, new_shape=imgsz)[0]
        # Convert
        img_letter = img_letter.transpose(2, 0, 1).copy()  # To 3x416x416

        img_letter = torch.from_numpy(img_letter).to(self.device)
        img_letter = img_letter.float()  # uint8 to fp16/32
        img_letter /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_letter.ndimension() == 3:
            img_letter = img_letter.unsqueeze(0)

        return img_letter

    @staticmethod
    def postprocess_dets(img, img_processed, preds):
        # Process detections
        # Original allows multiple frames to be processed in batch
        # In our case, we only process 1 image
        preds_img = preds[0]
        bboxes_probs = []
        landmarks = []

        if len(preds_img):
            # Rescale boxes from img_size to im0 size
            bboxes = scale_coords(img_processed.shape[2:], preds_img[:, :4], img.shape).round().cpu().numpy()
            probs = preds_img[:, 4].cpu().numpy().reshape(-1, 1)
            landmarks = scale_coords_landmarks(img_processed.shape[2:], preds_img[:, 5:15], img.shape).round().cpu().numpy()

            bboxes_probs = np.concatenate((bboxes, probs), axis=1)

        return bboxes_probs, landmarks

    def detect(self, img):
        # Preprocessing
        img_letter = self.preprocess_img(img)
        # Inference
        preds = self.model(img_letter)[0]
        # Apply NMS
        preds = non_max_suppression_face(preds, self.conf_thresh, self.nms_iou_thresh)
        # Postprocessing
        bboxes_probs, landmarks = self.postprocess_dets(img, img_letter, preds)

        return bboxes_probs, landmarks


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_one(model, image_path, device):
    import copy
    # Load model
    img_size = 693
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

    cv2.imwrite('result.jpg', orgimg)


if __name__ == "__main__":
    test_img = "data/faces_politicians/Yukio_Hatoyama/000001.jpeg"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo5_face = YOLO5FACE(device=DEVICE)
    detect_one(yolo5_face.model, test_img, DEVICE)
