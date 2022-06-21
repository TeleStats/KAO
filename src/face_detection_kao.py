# Useful resources: https://github.com/timesler/facenet-pytorch
import cv2
from itertools import cycle
import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
from time import time
from tqdm import tqdm
import torch

from config import get_parser
from model_classes import FeatureExtractor
from utils import skip_frames_until, read_rel_file
from utils import filter_path_list_by_date, convert_str_to_date, list_to_df_embeddings

# Common variables
# It's outside "__main__" due to multiprocessing not sharing memory of the elements inside __main__ for the "spawn" start method
# https://newbedev.com/workaround-for-using-name-main-in-python-multiprocessing
parser = get_parser()
args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {DEVICE}')

GT_POLITICIANS_PATH = Path('data/individuals')
VIDEOS_PATH = Path(f'data/{args.channel}')
DATASET_TRAIN_PATH = Path(f'data/dataset/train/{args.channel}')
# results_file = Path(f'data/results/2001_10_20_19_00.csv')
DET_PATH = None
FLAG_SAVE_VIDEO = args.save_video
FLAG_USE_DETS = args.use_dets
FLAG_EXTRACT_DETECTIONS = args.extract_dets
FLAG_EXTRACT_EMBEDDINGS = args.extract_embs
FROM_DATE = convert_str_to_date(args.from_date)
TO_DATE = convert_str_to_date(args.to_date)

if FLAG_USE_DETS:
    DET_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + 'detections')

if FLAG_EXTRACT_DETECTIONS:
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + 'detections')
elif FLAG_EXTRACT_EMBEDDINGS:
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats)
else:
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.mod_feat)

# Set which model is the face detector
if args.detector.lower() == 'mtcnn':
    from model_classes import FaceDetectorMTCNN as FaceDetector
elif args.detector.lower() == 'dfsd':
    from model_classes import FaceDetectorDFSD as FaceDetector
elif args.detector.lower() == 'yolo':
    from model_classes import FaceDetectorYOLO5 as FaceDetector
elif args.detector.lower() == 'insight':
    from model_classes import FaceDetectorInsightFace as FaceDetector
else:
    print(f'Detector {args.detector} not implemented!')
    exit()

# Feature extractor model
FEAT_MODEL = args.feats
FEAT_INPUT_SIZE = 160

# Do the detection for a specific year
if len(args.year) > 0:
    YEARS_TO_TEST = args.year
else:
    YEARS_TO_TEST = [str(y) for y in range(2001, 2022)]


def main_from_train_dataset(video_path, results_file, device=DEVICE, flag_save_vid=True):
    # Here we want to keep the result formatting from videos to be able to easily compare to other metrics
    year = results_file.parent.name
    base_frame_path = DATASET_TRAIN_PATH / year
    frames_year_path = [x for x in base_frame_path.iterdir() if x.stem.split('-')[0] == video_path]

    # Initialize face detector
    face_detector = FaceDetector(res_path=results_file.parent, video_name=results_file.stem + '_result', device=device)
    feat_extractor = FeatureExtractor(individuals_path=GT_POLITICIANS_PATH, feat_extractor=FEAT_MODEL, device=device,
                                      face_detector=face_detector)
    face_detector.set_feat_extractor(feat_extractor)

    with open(results_file, 'w') as w_res_file:
        # First row:
        res_row = ','.join(['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID']) + '\n'
        w_res_file.write(res_row)

        for frame_path in frames_year_path:
            frame = cv2.imread(str(frame_path))
            frame_to_detect = int(frame_path.stem.split('-')[1])
            face_detector.detect_and_write_to_file(frame, frame_to_detect, w_res_file, results_file,
                                                   flag_save_vid=flag_save_vid, flag_only_detections=FLAG_EXTRACT_DETECTIONS, flag_extract_embs=FLAG_EXTRACT_EMBEDDINGS)

        if FLAG_EXTRACT_EMBEDDINGS:
            df_embs = list_to_df_embeddings(face_detector.embs_list)
            df_embs.to_pickle(f'{RES_PATH}/{year}/{results_file.stem}.pkl')


def main_no_rel(video_path, results_file, sample_fps=1, device=DEVICE, flag_save_vid=True):
    cap = cv2.VideoCapture(str(video_path))
    v_width, v_height, v_fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    print(v_width, v_height, v_fps)

    year = results_file.parent.name
    if FLAG_USE_DETS:
        det_video_path = DET_PATH / year / results_file.name
    else:
        det_video_path = None

    # Initialize face detector
    face_detector = FaceDetector(det_path=det_video_path, res_path=results_file.parent,
                                 video_name=results_file.stem + '_result', video_fps=sample_fps, device=device)
    feat_extractor = FeatureExtractor(individuals_path=GT_POLITICIANS_PATH, feat_extractor=FEAT_MODEL, device=device,
                                      face_detector=face_detector)
    face_detector.set_feat_extractor(feat_extractor)

    vid_test_id = '_'.join(video_path.stem.split('_'))[:16]
    print(f'Video ID: {vid_test_id}')

    with open(results_file, 'w') as w_res_file:
        init_frame = 0
        end_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_to_detect = np.arange(init_frame, end_frame, v_fps)

        # First row:
        res_row = ','.join(['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID']) + '\n'
        w_res_file.write(res_row)

        # Loop over the whole video
        for frame_float in tqdm(frames_to_detect):
            frame_to_detect = int(frame_float)
            # frame = video[frame_to_detect]
            _ = skip_frames_until(cap, frame_to_detect)

            _, frame = cap.read()
            face_detector.detect_and_write_to_file(frame, cap.get(cv2.CAP_PROP_POS_FRAMES), w_res_file, results_file,
                                                   flag_save_vid=flag_save_vid, flag_only_detections=FLAG_EXTRACT_DETECTIONS, flag_extract_embs=FLAG_EXTRACT_EMBEDDINGS)

        if FLAG_EXTRACT_EMBEDDINGS:
            df_embs = list_to_df_embeddings(face_detector.embs_list)
            df_embs.to_pickle(f'{RES_PATH}/{year}/{results_file.stem}.pkl')


def main_train_mp(results_path, video_path, device=DEVICE):
    results_file = results_path / f'{video_path}.csv'
    main_from_train_dataset(video_path, results_file, flag_save_vid=FLAG_SAVE_VIDEO, device=device)


def main_demo_mp(results_path, video_path, device=DEVICE):
    # if not video_path.stem == '2011_04_10_19_00':
    #     return
    # In hodost-lv the folder and the video name are not the same
    video_file = [v for v in video_path.iterdir() if v.suffix == '.mp4'][0]
    # video_file = [v for v in video_path.iterdir() if v.suffix == '.mp4'][0]
    results_file = results_path / Path(f'{video_file.stem}.csv')
    main_no_rel(video_file, results_file, sample_fps=1, flag_save_vid=FLAG_SAVE_VIDEO, device=device)


def main_train():
    # Testing for the videos in the GT. Videos in per610a, GT in per920a
    time_init = time()
    # Check CUDA devices available
    if str(DEVICE) != 'cpu':
        cuda_devices_num = torch.cuda.device_count()
        devices = cycle([f'cuda:{i}' if args.process > 0 else f'cuda:{0}' for i in range(cuda_devices_num)])  # Circular list
    else:
        devices = cycle(['cpu'])

    for year_str in YEARS_TO_TEST:
        res_folder = Path(year_str)
        results_path = RES_PATH / res_folder
        videos_year_path = DATASET_TRAIN_PATH / year_str
        if videos_year_path.is_dir():
            os.makedirs(str(results_path), exist_ok=True)
        else:
            continue

        # Get unique values to get the video names of the annotated frames
        videos_year_day_path = list(set([x.stem.split('-')[0] for x in videos_year_path.iterdir() if x.suffix == '.jpg']))
        param_list = [(results_path, video_file, device) for (video_file, device) in zip(videos_year_day_path, devices)]
        if args.process > 0:
            # To erase processes from GPU memory
            # https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
            for i in range(0, len(param_list), args.process):
                with mp.Pool(processes=args.process) as p:
                    p.starmap(main_train_mp, param_list[i:i+args.process])
        else:
            for param in param_list:
                main_train_mp(param[0], param[1], param[2])

    print(f'Lasted {time() - time_init} seconds.')
    # 1 process: Around 8000 seconds --> 2:30 hours
    # 6 processes: Around 1600 seconds  --> 26 minutes


def main_demo():
    # Testing for the videos in the dataset (no GT). Videos in tv_news
    time_init = time()
    # Check CUDA devices available
    if str(DEVICE) != 'cpu':
        cuda_devices_num = torch.cuda.device_count()
        devices = cycle([f'cuda:{i}' if args.process > 0 else f'cuda:{0}' for i in range(cuda_devices_num)])   # Circular list
    else:
        devices = cycle(['cpu'])

    for year_str in YEARS_TO_TEST:
        res_folder = Path(year_str)
        results_path = RES_PATH / res_folder
        videos_year_path = VIDEOS_PATH / year_str
        if videos_year_path.is_dir():
            os.makedirs(str(results_path), exist_ok=True)
        else:
            continue

        videos_year_day_path = [x for x in videos_year_path.iterdir() if x.is_dir()]
        videos_year_day_path = filter_path_list_by_date(videos_year_day_path, FROM_DATE, TO_DATE)

        param_list = [(results_path, video_file, device) for (video_file, device) in zip(videos_year_day_path, devices)]
        if args.process > 0:
            for i in range(0, len(param_list), args.process):
                with mp.Pool(processes=args.process) as p:
                    p.starmap(main_demo_mp, param_list[i:i+args.process])
        else:
            for param in param_list:
                main_demo_mp(param[0], param[1], param[2])

    print(f'Lasted {time() - time_init} seconds.')
    # MTCNN (4 processes) 2001-2009 124067 seconds --> 34.5 hours
    # DFSD (3 processes [memory, slower(?)]) --> 142 hours


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if args.mode.lower() == 'demo':
        # Full videos
        main_demo()
    elif args.mode.lower() == 'train':
        # Frames inside the dataset/train folder (no video)
        main_train()
