from helper_functions import *
import random as rd
from tqdm import tqdm
from feature_extractor import get_hog_features_function


def sliding_window_for_hog_features(window_size, img, sliding_step, orientation, pixel_per_cell, cell_per_block):
    """
    SLide a window_size*window_size window in a image, with a sliding step, and then get HOG features of each
    window
    """
    img_init = img
    h, w = img_init.shape[0], img_init.shape[1]
    Features = []
    for x in range(0, h - window_size, sliding_step):
        for y in range(0, w - window_size, sliding_step):
            img = img_init[x:x + window_size, y:y + window_size, :]
            fd = get_hog_features_function(img, orientation, pixel_per_cell, cell_per_block, False)
            Features.append(fd)
    return Features


def extract_car_features(img, annotations, window_size, filling_factor, orientation, pixel_per_cell,
                         cell_per_block):
    """
    Extract image features of a image
    """
    img_init = img
    for annotation in annotations:
        img = img_init[annotation[1]:annotation[1] + annotation[3], annotation[0]:annotation[0] + annotation[2], :]

        if img.shape[0] > window_size or img.shape[1] > window_size:
            x_step = (img.shape[0] - window_size) // filling_factor
            y_step = (img.shape[1] - window_size) // filling_factor
            sliding_step = max(1, min(x_step, y_step))
            car_features = sliding_window_for_hog_features(window_size, img, sliding_step, orientation, pixel_per_cell,
                                                           cell_per_block)
            return car_features
        else:
            return None


def extract_non_car_features(annotations, img, number_of_boxes, box_size=64, orientation=8, pixel_per_cell=6,
                             cell_per_block=1):
    """
    Randomly pick number_of_boxes boxes which do not overlap with car frames
    """
    # non_cars_frames = []
    non_cars_features = []
    n_boxs = 0
    h, w = img.shape[0], img.shape[0]
    while n_boxs <= number_of_boxes:
        x = rd.randint(0, w - box_size - 1)
        y = rd.randint(0, h - box_size - 1)
        box = [(x, y), (x + box_size, y + box_size)]
        is_overlapping = False
        for annotation in annotations:
            box_car = transform_to_bbox(annotation)
            if is_overlap(box_car, box):
                is_overlapping = True
                break
        if not is_overlapping:
            n_boxs += 1
            # non_cars_frames.append(transform_to_annotation(box))
            img_non_car = img[y:y + box_size, x:x + box_size, :]
            fd = get_hog_features_function(img_non_car, orientation, pixel_per_cell, cell_per_block, False)
            non_cars_features.append(fd)
    return non_cars_features


def predict(model, img, window_size, sliding_step, confidence_threshold):
    """
    Predict and
    :param model: model already trained
    :param img: image where we want to detect cars
    :param window_size: the window size of the sliding window
    :param sliding_step: step at which we slide the window on the  image
    :param confidence_threshold: confidence threshold at which we consider a prediction
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    pred = []
    for x in tqdm(range(0, h - window_size, sliding_step)):
        for y in range(0, w - window_size, sliding_step):
            sliding_img = img[x:x + window_size, y:y + window_size, :]
            fd = get_hog_features_function(sliding_img, 8, 16, 1, False)
            score = model.decision_function([fd])[0]
            y_pred = model.predict([fd])
            if y_pred == 1 and score >= confidence_threshold:
                pred.append([x, y, x + window_size, y + window_size])
    return pred
