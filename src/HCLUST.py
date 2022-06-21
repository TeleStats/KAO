# Class for the HCLUST (Hierachical Clustering)
# Before using this extract features with extract_model_embeddings.py and from videos with face_detection_politics.py
import os

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as distance
from time import time
import torch

from config import get_parser
from utils import convert_str_to_date, filter_path_list_by_date, choose_n_pols


class HCLUST:
    def __init__(self, df, hierarchy_method='average',  fuse_method='distance', flag_fuse_pols=False):
        pd.options.mode.chained_assignment = None  # default='warn'
        self.df = df
        self.labels = None
        self.hierarchy_method = hierarchy_method
        self.fuse_method = fuse_method
        self.linkage_matrix = None
        self.flag_fuse_pols = flag_fuse_pols

    def generate_str_dist_matrix(self):
        # Input: labels (politician names list) --> '-1' is a label specified for the detections
        # Output: index list (True for same, False for different) in the form of pdist distance matrix
        idx_list = []
        for i, pol in enumerate(self.labels, 1):
            if self.flag_fuse_pols:
                idx_list += [True if (p == pol or p == '-1') else False for p in self.labels[i:]]  # Fuse models
            else:
                idx_list += [True if p == '-1' else False for p in self.labels[i:]]  # Don't fuse models

        return np.array(idx_list)

    def generate_hierarchical_tree(self, df, thresh=0.3):
        self.labels = [k_pol for k_pol in df['ID'].to_list()]  # Take the surnames as labels
        X = np.asarray(df['emb'].to_list())
        # Case where we only have 1 sample per video (in train we have 1 case)
        if X.shape[0] < 2:
            return [0], -1

        # Compute distance between features
        dist_matrix = distance.pdist(X, 'cosine')
        # Force different politicians into different clusters
        same_pol_idx = self.generate_str_dist_matrix()
        if len(same_pol_idx) > 0:
            dist_matrix[~same_pol_idx] += 1000
        v_debug = distance.squareform(dist_matrix)
        # Build hierarchical tree for the specified window
        linkage_matrix = linkage(dist_matrix, metric='cosine', method=f'{self.hierarchy_method}')
        # Cluster based on the hierarchical tree
        cluster_window = fcluster(linkage_matrix, t=thresh, criterion=f'{self.fuse_method}')

        return cluster_window, linkage_matrix

    @staticmethod
    def assign_labels_to_detections(df, clusters):
        df_clust = df.copy()
        df_clust['clust_ID'] = clusters
        df_clust_det = df_clust[df_clust['ID'] == '-1']
        clust_dets = df_clust_det['clust_ID'].unique()

        # We are interested in the clusters formed by the detections and whether they correspond to any politician,
        # but also saving all the others for the video generation
        for clust_id in clust_dets:
            mask_clust = df_clust['clust_ID'] == clust_id
            ids = df_clust['ID'][mask_clust].unique()
            if len(ids) > 1:
                pol_id = [i for i in ids if i != '-1'][0]
                mask_clust_det = df_clust_det['clust_ID'] == clust_id
                df_clust_det['ID'][mask_clust_det] = pol_id

        mask_not_assigned = df_clust_det['ID'] == '-1'
        # df_clust_det = df_clust_det.drop(df_clust_det[mask_not_assigned].index)

        return df_clust_det

    @staticmethod
    def format_df_for_evaluation(df):
        # In:  index,ID,emb,frame,cx,cy,w,h,prob_det,clust_ID
        # Out: source,ID,frame,cx,cy,w,h,prob_det,dist_ID
        df_out = df.copy()
        df_out['dist_ID'] = 0.1
        df_out = df_out[['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID']]

        return df_out


def format_x_axis(ax):
    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
    majors = ax.xaxis.get_major_formatter().func.args[0]
    major_axis = []
    major_str = []

    for k in majors.keys():
        if (majors[k] in POLITICIANS) and (majors[k] not in major_str):
            major_axis.append(k)
            major_str.append(majors[k])

    ax.xaxis.set_major_locator(ticker.FixedLocator(major_axis))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(major_str))
    ax.tick_params(axis='x', bottom=True)


def plot_dendrogram(model, y_lim=0.5, **kwargs):
    fig, ax1 = plt.subplots()
    dendrogram(model, **kwargs)
    format_x_axis(ax1)
    plt.setp(ax1.get_xticklabels(), ha="right", rotation=60, size=6)
    plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.show()


def main_toy():
    X1 = np.asarray([[0, 0], [0, 1], [1, 1]])
    X2 = np.asarray([[1, 0], [2, 0], [2, 2]])
    c1 = np.expand_dims(X1.sum(axis=0) / 3, axis=0)
    c2 = np.expand_dims(X2.sum(axis=0) / 3, axis=0)
    X1_c2 = np.append(X1, c2, axis=0)
    X2_c1 = np.append(X2, c1, axis=0)
    l1 = linkage(X1_c2, metric='euclidean', method='complete')
    l2 = linkage(X2_c1, metric='euclidean', method='complete')
    plot_dendrogram(l1)
    plot_dendrogram(l2)


def main():
    df_pol = pd.read_pickle(MODELS_PKL_PATH)
    df_pol = choose_n_pols(df_pol, num=0)
    df_pol['frame'] = '-1'  # If not initialized treats frames as floats and mismatches on metrics

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

            # Joint embeddings from detections and politicians
            df = pd.concat((df_pol, df_emb)).reset_index()

            hclust = HCLUST(df, hierarchy_method=HIER_METHOD)
            cluster_window, linkage_matrix = hclust.generate_hierarchical_tree(df, thresh=0.3)
            df_res = hclust.assign_labels_to_detections(df, cluster_window)
            df_res = hclust.format_df_for_evaluation(df_res)
            df_res.to_csv(f'{results_year_path}/{pkl_path.stem}.csv', index=False)
            # plot_dendrogram(linkage_matrix, labels=hclust.labels, y_lim=1.5, color_threshold=0.5)


def debug_magface():
    with open('/home/agirbau/work/faces/MagFace/inference/extracted_faces/feat.list', 'r') as f:
        lines = f.readlines()

    row_list = []
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        if imgname.find('Koizumi') > -1 or imgname.find('Abe') > -1:
            k_pol = 'Junichiro_Koizumi' if imgname.find('Koizumi') > -1 else 'Shinzo_Abe'
            feats = [float(e) for e in parts[1:]]
            mag = np.linalg.norm(feats)
            emb = feats / mag
            row_list.append([k_pol, emb, mag])

    df = pd.DataFrame(data=row_list, columns=['ID', 'emb', 'mag'])
    hclust = HCLUST(df, flag_fuse_pols=True)
    cluster_window, linkage_matrix = hclust.generate_hierarchical_tree(df, thresh=0.3)
    plot_dendrogram(linkage_matrix, labels=hclust.labels, y_lim=1)


def main_example_with_models():
    df = pd.read_pickle(MODELS_PKL_PATH)
    # Debug
    # pols = ['Junichiro_Koizumi', 'Shinzo_Abe']
    # df = df[df['ID'].isin(pols)]
    # import seaborn as sns
    # aa = np.asarray(df['emb'].to_list())
    # sim_mat = np.dot(aa, aa.T) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(aa, axis=1))
    # sns.heatmap(sim_mat, annot=True)
    # plt.show()
    hclust = HCLUST(df, flag_fuse_pols=True)
    cluster_window, linkage_matrix = hclust.generate_hierarchical_tree(df, thresh=0.3)
    plot_dendrogram(linkage_matrix, labels=hclust.labels, y_lim=1)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f'Mode:{args.mode}  Channel:{args.channel}  Detector:{args.detector}  Features:{args.feats}  Classifier:{args.mod_feat}')
    MODELS_PKL_PATH = Path(f'data/resources/models/{args.detector}-{args.feats}/model_embeddings.pkl')
    EMB_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats)
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats + '-' + args.mod_feat)
    HIER_METHOD = args.mod_feat.split('_')[-1]
    FROM_DATE = convert_str_to_date(args.from_date)
    TO_DATE = convert_str_to_date(args.to_date)

    init_time = time()
    # debug_magface()
    main()
    # main_example_with_models()
    # main_toy()
    print(f'Elapsed time: {round(time() - init_time, 1)}')
