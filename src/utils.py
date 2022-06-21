from collections import Counter
import csv

import copy
import cv2
import datetime
import json
import numpy as np
import os
from pathlib import Path

import pandas
import pandas as pd
from PIL import Image


def crop_face_images(img, bboxes):
    # Gets a pillow-type image and the bounding boxes as numpy array and returns pillow images cropped to the bboxes
    faces_img = []
    for bbox in bboxes:
        faces_img.append(img.crop(bbox))

    return faces_img


def crop_resize(img, box, output_size, resize_ok=True, bbox_air=0):
    # https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L309
    if isinstance(img, np.ndarray):
        p_top = max(0, box[1] - bbox_air)
        p_bot = min(box[3] + bbox_air, img.shape[0])
        p_left = max(0, box[0] - bbox_air)
        p_right = min(box[2] + bbox_air, img.shape[1])
        img = img[p_top:p_bot, p_left:p_right]
        if resize_ok:
            out = cv2.resize(
                img,
                (output_size, output_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        else:
            out = img.copy()
    else:
        if resize_ok:
            out = img.crop(box).copy().resize((output_size, output_size), Image.BILINEAR)
        else:
            out = img.crop(box).copy()

    return out


def bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    bbox_c_w = [cx, cy, w, h]
    return bbox_c_w


def bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox):
    x1 = bbox[0] - (bbox[2]/2)
    y1 = bbox[1] - (bbox[3]/2)
    x2 = bbox[0] + (bbox[2]/2)
    y2 = bbox[1] + (bbox[3]/2)

    bbox_x_y = [x1, y1, x2, y2]
    return bbox_x_y


def cm_to_inch(value):
    return value/2.54


def skip_frames_until(cap, frame_num):
    # Some videos do set correctly the frame with video[frame_num] or cap.set(2, frame_num), see https://github.com/opencv/opencv/issues/20227
    # Care with cv2 bugs (sometimes detects more frames in the video than the real ones, probably due to headers?)
    prev_frame = -1
    while True:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_num == current_frame:
            return 0
        elif frame_num < current_frame or prev_frame == current_frame:
            # Remember that VideoCapture.set() doesn't set at the 1st frame --> close and reopen
            # It has to be reopened outside, if not weird things happen
            return -1
        else:
            prev_frame = current_frame
            cap.grab()


def plot_bbox_label(bbox_xy, label, img, color, text_color=(0, 0, 0)):
    # Bounding box info
    x1 = int(bbox_xy[0])
    y1 = int(bbox_xy[1])
    x2 = int(bbox_xy[2])
    y2 = int(bbox_xy[3])
    # Take only surname for better visualisation
    label_plot = label.split('_')[-1]

    # Plot bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Plot label having a rectangle as background
    # From: https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label_plot, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Prints the text.
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label_plot, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


def fixed_image_standardization(image_tensor):
    # From MTCNN
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def choose_n_pols(df, num=0):
    # Function to test the influence of number of images of politicians in classification
    if num > 0:
        df_samples = df.groupby(["ID"]).sample(n=num)
    else:
        df_samples = df
    return df_samples


def get_gt_embeddings(face_detector, feat_extractor, models_path):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import my_transforms

    from face_dataset import FaceDataset

    # Dataloader
    transform = transforms.Compose([
        my_transforms.ToTensor()
    ])

    face_dataset = FaceDataset(data_path=models_path, transform=transform)
    face_dataloader = DataLoader(face_dataset, batch_size=2, shuffle=False, collate_fn=face_dataset.collate_fn)

    # Model already as a parameter
    # Face embeddings of the dataset per politician
    face_imgs = []
    face_embeddings = torch.Tensor().to(face_detector.device)
    face_labels = []

    for i, batch in enumerate(face_dataloader):
        imgs, labels = batch[0], batch[1]
        for img, label in zip(imgs, labels):
            img_pil = transforms.ToPILImage()(img).convert('RGB')
            bboxes, probs = face_detector.detect(img_pil)
            max_prob_idx = np.argmax(probs)  # In case there are 2 detections (Taro Aso in MTCNN)
            bboxes, probs = np.expand_dims(bboxes[max_prob_idx], axis=0), np.expand_dims(probs[max_prob_idx], axis=0)
            # Get face embeddings
            face_images_pil = crop_face_images(img_pil, bboxes)
            # Embeddings info
            [face_imgs.append(face_img_pil) for face_img_pil in face_images_pil]
            face_embeddings = torch.cat((face_embeddings, feat_extractor.crop_and_get_embeddings(img_pil, bboxes)))
            face_labels.append(label)

    return face_imgs, face_embeddings, face_labels


def add_padding(img, top, bot, left, right):
    # Add padding to an image
    # In case the input is a pillow image
    img_array = np.array(img)

    img_pad = cv2.copyMakeBorder(img_array, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Reconvert to pillow image
    if type(img) == Image.Image:
        img_pad = Image.fromarray(img_pad)

    return img_pad


def unpad_bbox(bbox, top, bot, left, right):
    # bbox: x1, y1, x2, y2
    bbox_unpad = copy.copy(bbox)
    bbox_unpad[0] -= left
    bbox_unpad[1] -= top
    bbox_unpad[2] -= right
    bbox_unpad[3] -= bot

    return bbox_unpad


def dict_to_df_custom(pol_dict):
    import pandas as pd

    df_dict = dict()
    df_dict['ID'] = []
    df_dict['emb'] = []
    for k_pol in pol_dict.keys():
        for emb in pol_dict[k_pol]['emb']:
            df_dict['ID'].append(k_pol)
            df_dict['emb'].append(emb.detach().cpu().numpy())

    df = pd.DataFrame(data=df_dict)
    return df


def list_to_df_embeddings(frame_embs_list: list) -> pd.DataFrame:
    # Function to transform the frame-embeddings tuple to a dataframe
    import pandas as pd

    df_dict = dict()
    # source,ID,frame,cx,cy,w,h,prob_det,emb
    source_list = [f_e[0] for f_e in frame_embs_list]
    id_list = [f_e[1] for f_e in frame_embs_list]
    frame_list = [int(f_e[2]) for f_e in frame_embs_list]
    cx_list = [f_e[3] for f_e in frame_embs_list]
    cy_list = [f_e[4] for f_e in frame_embs_list]
    w_list = [f_e[5] for f_e in frame_embs_list]
    h_list = [f_e[6] for f_e in frame_embs_list]
    prob_list = [f_e[7] for f_e in frame_embs_list]
    emb_list = [f_e[8] for f_e in frame_embs_list]

    df_dict['source'] = source_list
    df_dict['ID'] = id_list
    df_dict['frame'] = frame_list
    df_dict['cx'] = cx_list
    df_dict['cy'] = cy_list
    df_dict['w'] = w_list
    df_dict['h'] = h_list
    df_dict['prob_det'] = prob_list
    df_dict['emb'] = emb_list
    df = pd.DataFrame(df_dict)

    return df


def read_rel_file(rel_file):
    with open(rel_file, 'r') as json_file:
        rel_dict = json.load(json_file)

    return rel_dict


def convert_str_to_date(date_str: str) -> datetime.datetime:
    date_split = list(map(int, date_str.split('_')))
    yy = 2000
    mm = 1
    dd = 1
    if len(date_split) >= 1:
        yy = date_split[0]
    if len(date_split) >= 2:
        mm = date_split[1]
    if len(date_split) >= 3:
        dd = date_split[2]

    return datetime.datetime(yy, mm, dd)


def filter_list_by_date(str_dates_list: list, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    dates_list = [convert_str_to_date(x.split('-')[0]) for x in str_dates_list]  # split for csv train files (after '-' there's the frame number)
    idx_dates = [from_date.date() <= dd.date() <= to_date.date() for dd in dates_list]
    return idx_dates


def filter_path_list_by_date(path_dates_list: list, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    # Filter which videos to process
    videos_year_day_path_name = [vv.stem for vv in path_dates_list]
    idx_videos = filter_list_by_date(videos_year_day_path_name, from_date, to_date)
    videos_year_day_path_list = [dd for dd, bb in zip(path_dates_list, idx_videos) if bb]
    return videos_year_day_path_list


def get_same_els_list(list1: list, list2: list) -> list:
    union_set = set(list1) & set(list2)
    idxs = [list1.index(pol) for pol in union_set]
    return idxs


def get_video_from_path(video_path: Path, ext='.mpg'):
    # This needs to exists due to mismatch between folder path and video name in hodost-lv
    videos_in_path = [video_path for video_path in video_path.iterdir() if video_path.suffix == ext]
    if len(videos_in_path) > 1:
        print(f'Care, found more than one video: {videos_in_path}, taking {videos_in_path[0]} (first by default).')

    return videos_in_path[0]


def get_path_from_video(video_path: Path, ext='.mpg'):
    # I messed up naming the source information inside the dataframe (used video instead of folder info). For now:
    # This needs to exists due to mismatch between folder path and video name in hodost-lv
    date_video = video_path.stem[:10]
    folder_in_path = [folder_path for folder_path in video_path.parent.iterdir() if folder_path.stem.find(date_video) > -1]
    if len(folder_in_path) > 1:
        print(f'Care, found more than one video: {folder_in_path}, taking {folder_in_path[0]} (first by default).')

    return folder_in_path[0]


def get_most_repeated_value(in_list: list) -> list:
    # If 2 or more values are the most repeated return everything
    # From https://moonbooks.org/Articles/How-to-find-the-element-of-a-list-with-the-maximum-of-repetitions-occurrences-in-python-/
    out_list = Counter(in_list).most_common()
    return out_list


def drop_df_rows(df, col_id, rows_drop):
    # e.g. drop df['ID'] == 'Overall'
    if not isinstance(rows_drop, list):
        rows_drop_list = [rows_drop]
    else:
        rows_drop_list = rows_drop

    df_res = df.copy()
    for row_drop in rows_drop_list:
        mask_drop = df[col_id] == row_drop
        df_res = df_res[~mask_drop]

    return df_res


def read_and_append_df_csv(df, csv_path):
    if os.path.isfile(csv_path):
        df_csv = pd.read_csv(csv_path)
        df_res = pd.concat([df_csv, df])
    else:
        df_res = df

    return df_res
