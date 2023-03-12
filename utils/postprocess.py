import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from helper_functions import run_length_encoding, bounding_boxes_to_mask


def NMS(boxes, overlapThresh=0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    # print("computing overlap within all boxes")
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        box = np.array(box)
        xx1 = np.maximum(box[0], boxes[temp_indices, 0])
        yy1 = np.maximum(box[1], boxes[temp_indices, 1])
        xx2 = np.minimum(box[2], boxes[temp_indices, 2])
        yy2 = np.minimum(box[3], boxes[temp_indices, 3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual bounging box has an overlap bigger than treshold with any other box, remove it's index
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    # return only the boxes at the remaining indices
    return boxes[indices].astype(int)


def concatenate_predictions(predicted_cars_64, predicted_cars_128, is_NMS, N_test):
    predicted_cars_df = []
    for k in tqdm(range(N_test)):
        temp = predicted_cars_64[k] + predicted_cars_128[k]
        if is_NMS:
            temp = NMS(temp)
        predicted_cars_df.append(temp)
    return predicted_cars_df


def write_submission_file(predicted_cars_df, filename):
    H, W = 720, 1280
    test_files = sorted(os.listdir('./test/'))
    rows = []
    for k, file_name in enumerate(test_files):
        bounding_boxes = predicted_cars_df[k]
        bounding_boxes_adapted = [[elem[1], elem[0], elem[2] - elem[0], elem[3] - elem[1]] for elem in bounding_boxes]
        rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes_adapted, H, W))
        rows.append(['test/' + file_name, rle])

    df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
    df_prediction.to_csv(filename)
    print("Submission saved successfully")
    return None
