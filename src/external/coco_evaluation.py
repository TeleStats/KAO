# COCO evaluation tool extracted from https://github.com/rafaelpadilla/review_object_detection_metrics.git
from collections import defaultdict
from enum import Enum
from math import isclose
import numpy as np


# Bounding box formatting for COCO evaluation (starting after BoundingBox class)
class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BoundingBox:
    """ Class representing a bounding box. """
    def __init__(self,
                 image_name,
                 class_id=None,
                 coordinates=None,
                 type_coordinates=CoordinatesType.ABSOLUTE,
                 img_size=None,
                 bb_type=BBType.GROUND_TRUTH,
                 confidence=None,
                 format=BBFormat.XYWH):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
        """

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise IOError(
                'For bb_type=\'Detected\', it is necessary to inform the confidence value.')
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        # If YOLO format (rel_x_center, rel_y_center, rel_width, rel_height), change it to absolute format (x,y,w,h)
        if format == BBFormat.YOLO:
            assert self._width_img is not None and self._height_img is not None
            self._format = BBFormat.XYWH
            self._type_coordinates = CoordinatesType.RELATIVE

        self.set_coordinates(coordinates,
                             img_size=img_size,
                             type_coordinates=self._type_coordinates)

    def set_coordinates(self, coordinates, type_coordinates, img_size=None):
        self._type_coordinates = type_coordinates
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (type_coordinates == CoordinatesType.RELATIVE):
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if self._format == BBFormat.XYWH:
                (self._x, self._y, self._w,
                 self._h) = convert_to_absolute_values(img_size, coordinates)
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            elif self._format == BBFormat.XYX2Y2:
                x1, y1, x2, y2 = coordinates
                # Converting to absolute values
                self._x = round(x1 * self._width_img)
                self._x2 = round(x2 * self._width_img)
                self._y = round(y1 * self._height_img)
                self._y2 = round(y2 * self._height_img)
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = coordinates[0]
            self._y = coordinates[1]
            if self._format == BBFormat.XYWH:
                self._w = coordinates[2]
                self._h = coordinates[3]
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # self._format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = coordinates[2]
                self._y2 = coordinates[3]
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """ Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None):
        """ Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if img_size is not None:
            return convert_to_relative_values((img_size[0], img_size[1]),
                                              (self._x, self._x2, self._y, self._y2))
        else:
            return convert_to_relative_values((self._width_img, self._height_img),
                                              (self._x, self._x2, self._y, self._y2))

    def get_image_name(self):
        """ Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """ Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    def get_format(self):
        """ Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def get_class_id(self):
        """ Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """ Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    def get_area(self):
        assert isclose(self._w * self._h, (self._x2 - self._x) * (self._y2 - self._y))
        assert (self._x2 > self._x)
        assert (self._y2 > self._y)
        return (self._x2 - self._x + 1) * (self._y2 - self._y + 1)

    def get_coordinates_type(self):
        """ Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """ Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return f'image name: {self._image_name}\nclass: {self._class_id}\nbb (XYWH): {abs_bb_xywh}\nbb (X1Y1X2Y2): {abs_bb_xyx2y2}\narea: {area}\nbb_type: {self._bb_type}'

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            # unrelated types
            return False
        return str(self) == str(other)

    @staticmethod
    def compare(det1, det2):
        """ Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBox
            BoundingBox object representing one bounding box.
        dete2 : BoundingBox
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        det1BB = det1.getAbsoluteBoundingBox()
        det1img_size = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2img_size = det2.getImageSize()

        if det1.get_class_id() == det2.get_class_id() and \
           det1.get_confidence() == det2.get_confidence() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1img_size[0] == det1img_size[0] and \
           det2img_size[1] == det2img_size[1]:
            return True
        return False

    @staticmethod
    def clone(bounding_box):
        """ Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBox
            Cloned BoundingBox object.
        """
        absBB = bounding_box.get_absolute_bounding_box(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        new_bounding_box = BoundingBox(bounding_box.get_image_name(),
                                       bounding_box.get_class_id(),
                                       absBB[0],
                                       absBB[1],
                                       absBB[2],
                                       absBB[3],
                                       type_coordinates=bounding_box.getCoordinatesType(),
                                       img_size=bounding_box.getImageSize(),
                                       bb_type=bounding_box.getbb_type(),
                                       confidence=bounding_box.getConfidence(),
                                       format=BBFormat.XYWH)
        return new_bounding_box

    @staticmethod
    def iou(boxA, boxB):
        coords_A = boxA.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_B = boxB.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        # if boxes do not intersect
        if BoundingBox.have_intersection(coords_A, coords_B) is False:
            return 0
        interArea = BoundingBox.get_intersection_area(coords_A, coords_B)
        union = BoundingBox.get_union_areas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def get_intersection_area(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def get_union_areas(boxA, boxB, interArea=None):
        area_A = boxA.get_area()
        area_B = boxB.get_area()
        if interArea is None:
            interArea = BoundingBox.get_intersection_area(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def get_amount_bounding_box_all_classes(bounding_boxes, reverse=False):
        classes = list(set([bb._class_id for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(BoundingBox.get_bounding_box_by_class(bounding_boxes, c))
        # Sort dictionary by the amount of bounding boxes
        ret = {k: v for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)}
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes, class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(bounding_boxes, image_name):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return (round(x1), round(y1), round(x2), round(y2))


# COCO evaluation
""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""


def get_coco_summary(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])

    small = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in iou_thresholds
    }
    APsmall = [x['AP'] for k in small for x in small[k] if x['AP'] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    ARsmall = [
        x['TP'] / x['total positives'] for k in small for x in small[k] if x['TP'] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in iou_thresholds
    }
    APmedium = [x['AP'] for k in medium for x in medium[k] if x['AP'] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    ARmedium = [
        x['TP'] / x['total positives'] for k in medium for x in medium[k] if x['TP'] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in iou_thresholds
    }
    APlarge = [x['AP'] for k in large for x in large[k] if x['AP'] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)
    ARlarge = [
        x['TP'] / x['total positives'] for k in large for x in large[k] if x['TP'] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])

    return {
        "AP": AP,
        "AP50": AP50,
        "AP75": AP75,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge
    }


# Return TP, FP, and FN for all metrics in order to do further evaluation.
def get_coco_metrics_tpfpfn(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    out_dict = {"source": [], "ID": [], "bbox": [], "dist_ID": [], "scores": [], "matched": [], "NP": [], "gt": []}

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )

        # Capture detection results (matched with GT or not) for TP and FP
        for det, sc, mt in zip(_bbs[img_id, class_id]["dt"], ev["scores"], ev["matched"]):
            # In principle upper-left width height, but probably we built this as cx-cy-h (not doing the re-conversion)
            cx, cy, w, h = det.get_absolute_bounding_box(format=BBFormat.XYWH)
            # Here to state that I know about this, but if we converted the GT the same way it doesn't matter
            # cx = (x + w) / 2
            # cy = (y + h) / 2
            bbox = [cx, cy, w, h]
            dist_ID = 1 - det.get_confidence()

            out_dict["source"].append(img_id)
            out_dict["ID"].append(class_id)
            out_dict["bbox"].append(bbox)
            out_dict["dist_ID"].append(dist_ID)
            out_dict["scores"].append(sc)
            out_dict["matched"].append(mt)
            out_dict["NP"].append(ev["NP"])  # NP = TP + FN
            out_dict["gt"].append(False)

        # Capture GT results (matched with detections or not) for FN
        for gt, mt in zip(_bbs[img_id, class_id]["gt"], ev["matched_gt"]):
            if mt:
                continue
            else:
                cx, cy, w, h = gt.get_absolute_bounding_box(format=BBFormat.XYWH)
                bbox = [cx, cy, w, h]
                dist_ID = 1 - det.get_confidence()

                out_dict["source"].append(img_id)
                out_dict["ID"].append(class_id)
                out_dict["bbox"].append(bbox)
                out_dict["dist_ID"].append(dist_ID)
                out_dict["scores"].append(1)
                out_dict["matched"].append(mt)
                out_dict["NP"].append(ev["NP"])  # NP = TP + FN
                out_dict["gt"].append(True)

    return out_dict


def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _jaccard(a, b):
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched_gt = [gt_idx in gtm for gt_idx in range(len(gt)) if not gt_ignore[gt_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "matched_gt": matched_gt, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }
