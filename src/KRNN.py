# KRNN stands for K-Required Nearest Neighbors
# Script to classify the embeddings based on the KRNN with an arbitrary K
# Before using this extract features with extract_model_embeddings.py and from videos with face_detection_politics.py

import copy
import os

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
import scipy.spatial.distance as distance
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, NearestNeighbors
from time import time
import torch

from config import get_parser
from model_classes import FaceDetectorMTCNN as FaceDetector
from utils import convert_str_to_date, filter_path_list_by_date, get_most_repeated_value


class KRNNSolver:
    def __init__(self, df_pol, k_elems=1, flag_adapt=False):
        pd.options.mode.chained_assignment = None  # default='warn'
        # Data
        self.df_pol = df_pol
        self.dict_pol_tensor = dict()
        # Variables
        self.k_elems = k_elems
        self.flag_adapt = flag_adapt
        self.min_dist = 0.3
        self.adapt_dist = 0.15
        # Initializations
        self._init_pol_dict_with_tensors_()

    def _init_pol_dict_with_tensors_(self):
        # Turn the politicians embeddings to tensors to be able to compute the cosine distance of 1 vs. many using torch
        for k_pol in self.df_pol['ID'].unique():
            mask_pol = self.df_pol['ID'] == k_pol
            df = self.df_pol[mask_pol]
            self.dict_pol_tensor[k_pol] = dict()
            self.dict_pol_tensor[k_pol]['emb'] = torch.Tensor(np.stack(df['emb']))

    def assign_labels_to_detections(self, df_dets):
        # This method tries to emulate the method inside model_classes.py
        if self.flag_adapt:
            self._init_pol_dict_with_tensors_()  # Adapt politician models per video, not globally

        df_res = df_dets.copy()
        df_res['dist_ID'] = 100
        df_res = df_res.drop(columns=['emb'])
        frames_video = df_dets['frame'].unique()
        for frame in frames_video:
            mask_frame = df_dets['frame'] == frame
            df = df_dets[mask_frame]
            labels = ['-1'] * df.shape[0]  # List creation in Python, lol
            emb_dist_list = [100] * df.shape[0]
            min_dist = 100
            res_emb_dist = 100
            # I'm sure this can be done faster, but it's not that slow
            # CAVEAT: I just realized this is not KNN, but a criteria to assign a point to a known cluster by means of
            # computing the distance between the new point and the points (known) inside the cluster. To be KNN we
            # should compute the distance between the point and ALL vectors in the feature space and compute the
            # minimum K distances to assign the new point to a certain class.
            for idx, emb_det in enumerate(df['emb'].to_list()):
                emb_det_tensor = torch.Tensor(emb_det).unsqueeze(0)
                k_pol_adapt = None
                for k_pol in self.dict_pol_tensor.keys():
                    emb_dist = 1 - torch.cosine_similarity(self.dict_pol_tensor[k_pol]['emb'], emb_det_tensor)
                    k_neighs = min(self.k_elems, len(self.dict_pol_tensor[k_pol]['emb']))
                    num_close_elems = sum(emb_dist < self.min_dist)
                    if num_close_elems >= k_neighs:
                        res_emb_dist = torch.mean(emb_dist)  # 2 neighbors with 0.2 and 0.2 should be closer to 0.15 0.3 (?)
                    else:
                        res_emb_dist = 100

                    if res_emb_dist < min_dist and res_emb_dist < 0.3:
                        min_dist = res_emb_dist
                        labels[idx] = k_pol
                        emb_dist_list[idx] = round(res_emb_dist.item(), 2)
                        if self.flag_adapt and min_dist < 0.15:
                            k_pol_adapt = copy.copy(k_pol)

                # Add embedding to politician model
                if k_pol_adapt is not None:
                    self.dict_pol_tensor[k_pol_adapt]['emb'] = torch.cat([self.dict_pol_tensor[k_pol_adapt]['emb'], emb_det_tensor])

            # source,ID,frame,cx,cy,w,h,prob_det,dist_ID
            df_res['ID'][mask_frame] = labels
            df_res['dist_ID'][mask_frame] = emb_dist_list

        return df_res


def main():
    df_pol = pd.read_pickle(MODELS_PKL_PATH)
    df_pol['frame'] = '-1'  # If not initialized treats frames as floats and mismatches on metrics
    knn_solver = KRNNSolver(df_pol, k_elems=int(K_ELEMS), flag_adapt=FLAG_ADAPT)

    years = [x.stem for x in EMB_PATH.iterdir() if x.is_dir()]
    for year in years:
        embs_year_path = EMB_PATH / year
        results_year_path = RES_PATH / year
        os.makedirs(results_year_path, exist_ok=True)

        emb_files_year = [x for x in embs_year_path.iterdir() if x.is_file() and x.suffix == '.pkl']
        emb_files_year = filter_path_list_by_date(emb_files_year, FROM_DATE, TO_DATE)

        for pkl_path in emb_files_year:
            # Debug
            # if pkl_path.stem != '2001_06_30_19_00':
            #     continue

            df_emb = pd.read_pickle(pkl_path)
            df_res = knn_solver.assign_labels_to_detections(df_emb)
            df_res.to_csv(f'{results_year_path}/{pkl_path.stem}.csv', index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f'Mode:{args.mode}  Detector:{args.detector}  Features:{args.feats}  Classifier:{args.mod_feat}')
    MODELS_PKL_PATH = Path(f'data/resources/models/{args.detector}-{args.feats}/model_embeddings.pkl')
    EMB_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats)
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats + '-' + args.mod_feat)
    # e.g. knn_1_adapt
    # Get minimum between the specified K and the amount of embeddings per politician
    K_ELEMS = args.mod_feat.split('_')[1]
    FLAG_ADAPT = True if args.mod_feat.find('adapt') > -1 else False
    FROM_DATE = convert_str_to_date(args.from_date)
    TO_DATE = convert_str_to_date(args.to_date)

    init_time = time()
    main()
    print(f'Elapsed time: {round(time() - init_time, 1)}')
