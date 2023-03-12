import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread



def read_frame(df_annotation, frame):
    """Read frames and create integer frame_id-s"""
    file_path = df_annotation[df_annotation.index == frame]['frame_id'].values[0]
    return imread(file_path)


def annotations_for_frame(df_annotation, frame):
    assert frame in df_annotation.index
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    if pd.isna(bbs):  # some frames contain no vehicles
        return []

    bbs = list(map(lambda x: int(x), bbs.split(' ')))
    return np.array_split(bbs, len(bbs) / 4)


def show_annotation(df_annotation, frame):
    img = read_frame(df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    fig, ax = plt.subplots(figsize=(10, 8))

    for x, y, dx, dy in bbs:
        rect = patches.Rectangle((x, y), dx, dy, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title('Annotations for frame {}.'.format(frame))


def bounding_boxes_to_mask(bounding_boxes, H, W):
    """
    Converts set of bounding boxes to a binary mask
    """

    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[y:y + dy, x:x + dx] = 1

    return mask


def run_length_encoding(mask):
    """
    Produces run length encoding for a given binary mask
    """

    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]

    if non_zeros.size == 0:
        return ''
    else:

        padded = np.pad(non_zeros, pad_width=1, mode='edge')

        # find start and end points of non-zeros runs
        limits = (padded[1:] - padded[:-1]) != 1
        starts = non_zeros[limits[:-1]]
        ends = non_zeros[limits[1:]]
        lengths = ends - starts + 1

        return ' '.join(['%d %d' % (s, l) for s, l in zip(starts, lengths)])

def transform_to_bbox(annotation):
    """
    Transform an annotation to a bounding box
    :param annotation:
    :return:
    """
    return [(annotation[0], annotation[1]), (annotation[0]+annotation[2], annotation[1]+annotation[3])]

def transform_to_annotation(box):
    """
    Transform a bounding box to annotation
    :param box:
    :return:
    """
    x1, y1 = box[0]
    x2, y2 = box[1]
    return np.array([x1, y1, x2-x1, y2-y1 ])

def is_overlap(box1, box2):
    """
    Check if 2 boxes are overlapping
    :param box1:
    :param box2:
    :return:
    """
    x1, y1 = box1[0]
    x2, y2 = box1[1]
    x3, y3 = box2[0]
    x4, y4 = box2[1]
    if x1 > x4 or x3 > x2:
        return False
    if y1 > y4 or y3 > y2:
        return False
    return True