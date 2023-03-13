from skimage.feature import hog




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


