import os

import pandas as pd
from pathlib import Path
from time import time

from config import get_parser
from HCLUST import HCLUST
from FCG import FCG
from KNN import KNNSolver
from KRNN import KRNNSolver
from utils import convert_str_to_date, filter_path_list_by_date, choose_n_pols


def main():
    df_pol = pd.read_pickle(MODELS_PKL_PATH)
    df_pol = choose_n_pols(df_pol, num=0)
    df_pol['frame'] = '-1'  # If not initialized treats frames as floats and mismatches on metrics
    df_res = None

    years = [x.stem for x in EMB_PATH.iterdir() if x.is_dir()]
    for year in years:
        embs_year_path = EMB_PATH / year
        results_year_path = RES_PATH / year
        os.makedirs(results_year_path, exist_ok=True)

        emb_files_year = [x for x in embs_year_path.iterdir() if x.is_file() and x.suffix == '.pkl']
        emb_files_year = filter_path_list_by_date(emb_files_year, FROM_DATE, TO_DATE)

        for pkl_path in emb_files_year:
            # Debug
            # if pkl_path.stem != '2021_05_28_21_54':
            #     continue
            # print(pkl_path.stem)

            df_emb = pd.read_pickle(pkl_path)
            if df_emb.empty:
                continue

            # Joint embeddings from detections and politicians
            df = pd.concat((df_pol, df_emb)).reset_index()

            if BASENAME == 'fcg':
                # e.g. fcg_average_vote
                hier_method = METHOD_ARGS[1]
                if len(METHOD_ARGS) > 2:
                    class_method = METHOD_ARGS[2]
                else:
                    class_method = 'vote'
                fcg = FCG(df_emb, hierarchy_method=hier_method, classify_method=class_method)
                cluster_window, linkage_matrix = fcg.generate_intra_clusters(df_emb, thresh=0.3)
                df_clust = fcg.assign_df_cluster_labels(df_emb, cluster_window)
                thresh = 0.7
                perc_votes = 0.5

                df_res = fcg.classify_clusters(df_pol, df_clust, thresh=thresh, perc_votes=perc_votes)
                df_res = fcg.format_df_for_evaluation(df_res)

            elif BASENAME == 'hclust':
                # e.g. hclust_average
                hier_method = METHOD_ARGS[-1]
                hclust = HCLUST(df, hierarchy_method=hier_method)
                cluster_window, linkage_matrix = hclust.generate_hierarchical_tree(df, thresh=0.3)
                df_res = hclust.assign_labels_to_detections(df, cluster_window)
                df_res = hclust.format_df_for_evaluation(df_res)

            elif BASENAME == 'knn':
                # e.g. knn_1_adapt
                # Get minimum between the specified K and the amount of embeddings per politician
                k_elems = METHOD_ARGS[1]
                flag_adapt = 'adapt' in METHOD_ARGS
                knn_solver = KNNSolver(df_pol, k_elems=int(k_elems), flag_adapt=flag_adapt)
                df_res = knn_solver.assign_labels_to_detections(df_emb)

            elif BASENAME == 'krnn':
                # e.g. krnn_1_adapt
                # Get minimum between the specified K and the amount of embeddings per politician
                k_elems = METHOD_ARGS[1]
                flag_adapt = 'adapt' in METHOD_ARGS
                krnn_solver = KRNNSolver(df_pol, k_elems=int(k_elems), flag_adapt=flag_adapt)
                df_res = krnn_solver.assign_labels_to_detections(df_emb)

            df_res.to_csv(f'{results_year_path}/{pkl_path.stem}.csv', index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f'Mode:{args.mode}  Channel:{args.channel}  Detector:{args.detector}  Features:{args.feats}  Classifier:{args.mod_feat}')
    MODELS_PKL_PATH = Path(f'data/resources/individuals/{args.detector}-{args.feats}/individuals_embeddings.pkl')
    EMB_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats)
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats + '-' + args.mod_feat)

    FROM_DATE = convert_str_to_date(args.from_date)
    TO_DATE = convert_str_to_date(args.to_date)

    METHOD_ARGS = args.mod_feat.split('_')
    BASENAME = METHOD_ARGS[0]

    init_time = time()
    main()
    print(f'Elapsed time: {round(time() - init_time, 1)}')
