import cv2
from skimage.feature import hog
from helper_functions import *
import random as rd



def get_hog_features_function(img, orientation, pixel_per_cell, cell_per_block, vis):
    """
    Extract HOG features from an image
    """
    if vis:
        fd, hog_image = hog(img, orientations=orientation, pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block), visualize=True, channel_axis=-1)
        return fd, hog_image
    else:
        fd = hog(img, orientations=orientation, pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                 cells_per_block=(cell_per_block, cell_per_block), visualize=False, channel_axis=-1)
    return fd


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


def extract_car_features_bis(img, annotations, window_size, filling_factor, orientation, pixel_per_cell,
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


def extract_car_features(img, annotations, orientation, cell_per_block):
    """
    Extract features from training set images
    :param img: image containing car features
    :param annotations: annotations of frames where cars are detected
    :param orientation: parameter of the HOG feature
    :param cell_per_block: parameter of the HOG feature
    :return: 2 lists of respectively features of size 64*64 and 128*128
    """
    img_init = img
    cars_64_64 = []
    cars_128_128 = []

    for annotation in annotations:
        img = img_init[annotation[1]:annotation[1] + annotation[3], annotation[0]:annotation[0] + annotation[2], :]
        if img.shape[0] >= 64 and img.shape[1] >= 64:
            pixel_per_cell = 6
            img_reshaped_64 = cv2.resize(img, (64, 64), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            fd = get_hog_features_function(img_reshaped_64, orientation, pixel_per_cell, cell_per_block, False)
            cars_64_64.append(fd)
            if img.shape[1] >= 128 or img.shape[0] >= 128:
                pixel_per_cell = 16
                img_reshaped_128 = cv2.resize(img, (128, 128), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                fd = get_hog_features_function(img_reshaped_128, orientation, pixel_per_cell, cell_per_block, False)
                cars_128_128.append(fd)

    return cars_64_64, cars_128_128


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
