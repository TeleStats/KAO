# Class for the FCG (Feature Combinational Grouping)
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

from config import get_parser
from utils import convert_str_to_date, filter_path_list_by_date, choose_n_pols


class FCG:
    def __init__(self, df, hierarchy_method='average',  fuse_method='distance', classify_method='vote'):
        pd.options.mode.chained_assignment = None  # default='warn'
        self.df = df
        self.labels = None
        self.hierarchy_method = hierarchy_method
        self.fuse_method = fuse_method
        self.classify_method = classify_method
        self.linkage_matrix = None

    def generate_intra_clusters(self, df, thresh=0.3):
        X = np.asarray(df['emb'].to_list())
        # Case where we only have 1 sample per video (in train we have 1 case)
        if X.shape[0] < 2:
            return [0], -1
        # Case where the video does not contain any detection (can happen in train data, for samples without faces)
        if len(X) == 0:
            return [0], -1

        # Compute distance between features
        dist_matrix = distance.pdist(X, 'cosine')
        v_debug = distance.squareform(dist_matrix)
        # Build hierarchical tree for the specified window
        # To better understand the linkage matrix check https://stackoverflow.com/questions/9838861/scipy-linkage-format
        linkage_matrix = linkage(dist_matrix, metric='cosine', method=f'{self.hierarchy_method}')
        # Cluster based on the hierarchical tree
        cluster_window = fcluster(linkage_matrix, t=thresh, criterion=f'{self.fuse_method}')

        return cluster_window, linkage_matrix

    @staticmethod
    def assign_df_cluster_labels(df, cluster_labels):
        # Input: Original df, cluster labels
        # Output: df with cluster labels
        df_cluster_labels = df.copy()
        df_cluster_labels['clust_ID'] = cluster_labels
        return df_cluster_labels

    @staticmethod
    def assign_label(X_models, X_clust, model_labels, method='centroid', thresh=0.7, perc_votes=0.5):
        # Return index of the label to further check the df
        label = '-1'
        dist = 1
        if method == 'centroid':
            # Compare models with centroid of the cluster (tracklet)
            X_centroid = X_clust.mean(axis=0, keepdims=True)
            dist_matrix = distance.cdist(X_models, X_centroid, 'cosine')
            label_idx = np.argmin(dist_matrix)
            if dist_matrix.min() < thresh:
                label = model_labels[label_idx]
                dist = dist_matrix[label_idx]

        elif method == 'vote':
            # Compare models with all the elements of the tracklet and do a majority vote (> 50%)
            dist_matrix = distance.cdist(X_models, X_clust, 'cosine')
            label_idx = np.argmin(dist_matrix, axis=0)
            dists = np.asarray([dist_matrix[mod_idx][det_idx] for det_idx, mod_idx in enumerate(label_idx)])
            labels = np.asarray([model_labels[mod_idx] for mod_idx in label_idx])
            dist_below_thresh = list(dists < thresh)
            labels_below_thresh = list(labels[dist_below_thresh])
            # Votes per model label (votes per politician)
            labels_unique = set(labels_below_thresh)
            votes_max = 0
            for l_unique in labels_unique:
                votes = labels_below_thresh.count(l_unique)
                dist_tmp = np.asarray([dists[i] for i, l in enumerate(labels) if l == l_unique]).mean()
                if votes >= votes_max and dist_tmp < dist:
                    votes_max = votes
                    if (votes/len(dists)) > perc_votes:
                        label = l_unique
                        dist = min(np.asarray([dists[i] for i, l in enumerate(labels) if l == l_unique]))

        return label, dist

    def classify_clusters(self, df_models, df_cluster_labels, thresh=0.7, perc_votes=0.5):
        # Try to match models with the generated tracklets
        # Input: df_models, df_cluster_labels
        # Output: df_cluster_labels with the assigned model (or -1 if no match)
        df_res = df_cluster_labels.copy()
        df_res['dist_ID'] = 1
        X_models = np.asarray(df_models['emb'].to_list())
        model_labels = df_models['ID'].to_list()
        for clust in df_cluster_labels['clust_ID'].unique():
            mask_clust = df_cluster_labels['clust_ID'] == clust
            embs = np.asarray(df_cluster_labels['emb'][mask_clust].to_list())
            label, dist = self.assign_label(X_models, embs, model_labels, method=self.classify_method, thresh=thresh, perc_votes=perc_votes)
            df_res['ID'][mask_clust] = label
            df_res['dist_ID'][mask_clust] = dist

        return df_res

    @staticmethod
    def format_df_for_evaluation(df):
        # In:  index,ID,emb,frame,cx,cy,w,h,prob_det,clust_ID
        # Out: source,ID,frame,cx,cy,w,h,prob_det,dist_ID
        df_out = df.copy()
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

            fcg = FCG(df_emb, hierarchy_method=HIER_METHOD, classify_method='vote')
            cluster_window, linkage_matrix = fcg.generate_intra_clusters(df_emb, thresh=0.3)
            df_clust = fcg.assign_df_cluster_labels(df_emb, cluster_window)
            df_res = fcg.classify_clusters(df_pol, df_clust)
            df_res = fcg.format_df_for_evaluation(df_res)
            df_res.to_csv(f'{results_year_path}/{pkl_path.stem}.csv', index=False)
            # plot_dendrogram(linkage_matrix, labels=fcg.labels, y_lim=1.5, color_threshold=0.5)


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
    fcg = FCG(df)
    cluster_window, linkage_matrix = fcg.generate_intra_clusters(df, thresh=0.3)
    plot_dendrogram(linkage_matrix, labels=fcg.labels, y_lim=1)


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
    main()
    # main_example_with_models()
    # main_toy()
    print(f'Elapsed time: {round(time() - init_time, 1)}')
