"""Implementation of various metrics for scoring boundary maps.

"""

import keras.backend as K

__all__ = ['precision', 'recall', 'f_score', 'pixel_error', 'rand_error', 'wrapping_error', 'iou', ]


def precision(y_true, y_pred):
    pass


def recall(y_true, y_pred):
    pass


def f_score(y_true, y_pred):
    pass


def pixel_error(y_true, y_pred):
    pass


def rand_error(y_true, y_pred):
    pass


def wrapping_error(y_true, y_pred):
    pass


def iou(y_true, y_pred):
    """
    Intersection over Union (IoU) metric, also known as the Jaccard index.
    Defined as intersection divided by union of the sample sets.

    IoU = |target and prediction| / |target or prediction|
        = |target| + |prediction| - |target or prediction| / |target or prediction|
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return union / intersection
