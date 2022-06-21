# Script to extract and save individual embeddings to not to recompute them every time
# Embedding file (pickle) will be saved inside "data/resources" as "model_embeddings.pkl"
# Run from parent path (KAO)
from pathlib import Path
import os
import torch

from config import get_parser
from model_classes import FeatureExtractor
from utils import dict_to_df_custom


def main():
    res_path = RES_PATH / 'individuals_embeddings.pkl'
    face_detector = FaceDetector(device=DEVICE)
    feat_extractor = FeatureExtractor(individuals_path=MODELS_PATH, feat_extractor=FEAT_MODEL, device=DEVICE,
                                      face_detector=face_detector)
    face_detector.set_feat_extractor(feat_extractor)

    df_embs = dict_to_df_custom(feat_extractor.individuals_dict)
    df_embs.to_pickle(f'{res_path}')


if __name__ == "__main__":
    # Common variables
    # It's outside "__main__" due to multiprocessing not sharing memory of the elements inside __main__ for the "spawn" start method
    # https://newbedev.com/workaround-for-using-name-main-in-python-multiprocessing
    parser = get_parser()
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {DEVICE}')

    MODELS_PATH = Path(f'data/individuals')
    RES_PATH = Path("data/resources/individuals") / Path(args.detector + '-' + args.feats)
    os.makedirs(RES_PATH, exist_ok=True)
    FEAT_MODEL = args.feats

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

    main()
