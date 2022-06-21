# Script that loads the results and maps them to the original video / images
import os

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import get_parser
from utils import convert_str_to_date, skip_frames_until, bbox_cx_cy_w_h_to_x1_y1_x2_y2, crop_resize, plot_bbox_label


class VideoGenerator:
    def __init__(self, root_res_path, root_dataset_path):
        # Constants
        self.root_res_path = root_res_path
        self.root_dataset_path = root_dataset_path
        # Variables
        self.df_res = None
        self.vid_path = None
        self.vid_res = None

    def load_results(self, res_path):
        self.df_res = pd.read_csv(res_path)
        year = res_path.parent.stem
        self.vid_path = self.root_dataset_path / year / res_path.stem / (res_path.stem + '.mp4')
        self.vid_res = res_path.with_suffix('.mp4')

    def _iterate_through_video_(self, cap, opt_skip=True):
        if opt_skip:
            frames = self.df_res['frame'].unique()
            for frame_num in frames:
                frame_to_extract = int(frame_num)
                _ = skip_frames_until(cap, frame_to_extract)
                _, img = cap.read()

                yield frame_to_extract, img
        else:
            flag_ok = True
            while flag_ok:
                frame_to_extract = cap.get(cv2.CAP_PROP_POS_FRAMES)
                flag_ok, img = cap.read()
                yield frame_to_extract, img

    def generate_video(self):
        cap = cv2.VideoCapture(str(self.vid_path))
        v_width, v_height, v_fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        print(v_width, v_height, v_fps)
        vid_writer = cv2.VideoWriter(str(self.vid_res), cv2.VideoWriter_fourcc(*'MP4V'), v_fps, (v_width, v_height))

        color = (255, 255, 255)
        cap_iterator = self._iterate_through_video_(cap, opt_skip=False)

        # The analysis has been done at 1 fps, keep the plotted bboxes that amount of time
        frame_plot_count = v_fps + 1
        bboxes_xy = []
        labels = []

        for frame_to_extract, img in tqdm(cap_iterator):
            df_frame = self.df_res[self.df_res['frame'] == frame_to_extract]

            if df_frame.empty:
                frame_plot_count += 1
                if frame_plot_count < v_fps:
                    for bbox_xy, label in zip(bboxes_xy, labels):
                        img = plot_bbox_label(bbox_xy, label, img, color)
            else:
                bboxes_xy = []
                labels = []

            for row in df_frame.iterrows():
                # Reset counter for plotting bounding boxes
                frame_plot_count = 0

                label = row[1]['ID']
                cx = row[1]['cx']
                cy = row[1]['cy']
                w = row[1]['w']
                h = row[1]['h']
                bbox_cwh = (cx, cy, w, h)
                bbox_xy = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_cwh)

                bboxes_xy.append(bbox_xy)
                labels.append(label)

                img = plot_bbox_label(bbox_xy, label, img, color)

            cv2.putText(img, str(frame_to_extract), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            vid_writer.write(img)

    def extract_faces(self):
        cap = cv2.VideoCapture(str(self.vid_path))
        cap_iterator = self._iterate_through_video_(cap)

        for frame_to_extract, img in cap_iterator:
            df_frame = self.df_res[self.df_res['frame'] == frame_to_extract]
            for row in df_frame.iterrows():
                label = row[1]['ID']
                cx = row[1]['cx']
                cy = row[1]['cy']
                w = row[1]['w']
                h = row[1]['h']
                bbox_cwh = (cx, cy, w, h)
                bbox_xy = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_cwh)
                bbox_xy = [int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])]
                face_img = crop_resize(img, bbox_xy, 64)

                face_path = Path(f'figures/faces/{self.vid_res.stem}/{label}-{frame_to_extract}.jpg')
                os.makedirs(face_path.parent, exist_ok=True)
                cv2.imwrite(str(face_path), face_img)


def main():
    video_generator = VideoGenerator(RES_PATH, DATASET_PATH)

    # res_path_test = RES_PATH / Path('2008/2008_12_21_19_00.csv')
    res_path_test = RES_PATH / Path('2016/2016_09_27_19_00.csv')
    video_generator.load_results(res_path_test)
    video_generator.generate_video()
    # video_generator.extract_faces()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    DATASET_PATH = Path(f'data/{args.channel}')
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats + '-' + args.mod_feat)

    FROM_DATE = convert_str_to_date(args.from_date)
    TO_DATE = convert_str_to_date(args.to_date)

    main()
