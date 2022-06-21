# Script to classify the embeddings based on the KNN with an arbitrary K
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
from utils import convert_str_to_date, filter_path_list_by_date, get_most_repeated_value, choose_n_pols


class KNNSolver:
    def __init__(self, df_pol, k_elems=1, flag_adapt=False):
        pd.options.mode.chained_assignment = None  # default='warn'
        # Data
        self.df_pol = df_pol
        self.dict_pol_array = dict()
        self.knn_nearest = None
        # Variables
        self.k_elems = k_elems
        self.flag_adapt = flag_adapt
        self.min_dist = 0.3
        self.adapt_dist = 0.15
        # Initializations
        self._init_pol_dict_with_arrays_()
        self._init_knn_()

    def _init_pol_dict_with_arrays_(self):
        # Turn the politicians embeddings to arrays to compute KNN (and also be able to adapt the KNN models)
        for k_pol in self.df_pol['ID'].unique():
            mask_pol = self.df_pol['ID'] == k_pol
            df = self.df_pol[mask_pol]
            self.dict_pol_array[k_pol] = dict()
            self.dict_pol_array[k_pol]['emb'] = np.stack(df['emb'])

    def _init_knn_(self):
        # Initialize KNN classifier
        # Cosine is not technically a distance, but euclidean should keep the same ordering
        # self.knn_classifier = KNeighborsClassifier(n_neighbors=self.k_elems, radius=self.min_dist, metric='cosine')
        self.knn_nearest = NearestNeighbors(n_neighbors=self.k_elems, radius=self.min_dist, metric='cosine')
        # Do this with array dictionary to be able to re-use it for adaptive KNN
        # X = self.df_pol['emb'].to_list()
        # y = self.df_pol['ID'].to_list()
        X = []
        y = []
        for label in self.dict_pol_array.keys():
            for emb in self.dict_pol_array[label]['emb']:
                X.append(emb)
                y.append(label)

        # self.knn_classifier.fit(X, y)
        self.knn_nearest.fit(X)
        self.knn_nearest.labels = y
        # self.knn_classifier.predict(np.reshape(X[0], (1, -1)))

    def assign_labels_to_detections(self, df_dets):
        # This method tries to emulate the method inside model_classes.py
        if self.flag_adapt:
            self._init_pol_dict_with_arrays_()  # Adapt politician models per video, not globally
            self._init_knn_()

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
            for idx, emb_det in enumerate(df['emb'].to_list()):
                emb_det_array = np.reshape(emb_det, (1, -1))
                # predicted_label = self.knn_classifier.predict(emb_det_array)
                dists, idx_labels = self.knn_nearest.kneighbors(emb_det_array)
                # Filter by min_dist
                dists_filtered = [d for d in dists[0] if d <= self.min_dist]
                idx_labels_filtered = [idx_labels[0][i] for i, d in enumerate(dists[0]) if d <= self.min_dist]
                if len(idx_labels_filtered) == 0:
                    continue

                # Check possible labels
                possible_labels = [self.knn_nearest.labels[int(idx_label)] for idx_label in idx_labels_filtered]
                count_labels = get_most_repeated_value(possible_labels)
                predicted_label = count_labels[0][0]
                idxs_label = [i for i, l in enumerate(possible_labels) if l == predicted_label]
                dists_predicted_label = [dists_filtered[i] for i in idxs_label]
                dist_mean = np.mean(dists_predicted_label)

                # Add label and distance to results
                labels[idx] = predicted_label
                emb_dist_list[idx] = round(dist_mean, 2)

                if self.flag_adapt and dist_mean < self.adapt_dist:
                    self.dict_pol_array[predicted_label]['emb'] = np.concatenate(([self.dict_pol_array[predicted_label]['emb'], emb_det_array]))
                    self._init_knn_()

            # source,ID,frame,cx,cy,w,h,prob_det,dist_ID
            df_res['ID'][mask_frame] = labels
            df_res['dist_ID'][mask_frame] = emb_dist_list

        return df_res


def main():
    df_pol = pd.read_pickle(MODELS_PKL_PATH)
    # df_pol = choose_n_pols(df_pol, num=2)  # For testing
    df_pol['frame'] = '-1'  # If not initialized treats frames as floats and mismatches on metrics
    knn_solver = KNNSolver(df_pol, k_elems=int(K_ELEMS), flag_adapt=FLAG_ADAPT)

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
            # df_res = knn_solver.assign_labels_to_detections(df_emb)
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
