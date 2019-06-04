"""Implementation of various metrics for scoring boundary maps.
"""

import keras.backend as K

__all__ = ['precision', 'recall', 'accuracy', 'f_score', 'pixel_error', 'rand_error', 'wrapping_error', 'dice', 'iou']


def dimension_check(y_true, y_pred):
    if not y_true.shape == y_pred.shape:
        raise ValueError("Dimensions of y_true and y_pred differ!")


def tp(y_true, y_pred):
    return K.sum(y_true * y_pred)


def fp(y_true, y_pred):
    return K.sum((1 - y_true) * y_pred)


def tn(y_true, y_pred):
    return K.sum((1 - y_true) * (1 - y_pred))


def fn(y_true, y_pred):
    return K.sum(y_true * (1 - y_pred))


def precision(y_true, y_pred):
    """TP / (TP + FP)
    """
    TP = tp(y_true, y_pred)
    FP = fp(y_true, y_pred)
    return TP / (TP + FP)


def recall(y_true, y_pred):
    """TP / (TP + FN)
    """
    TP = tp(y_true, y_pred)
    FN = fn(y_true, y_pred)
    return TP / (TP + FN)


def accuracy(y_true, y_pred):
    """(TP + TN) / (TP + FP + TN + FN)
    """
    TP = tp(y_true, y_pred)
    TN = tn(y_true, y_pred)
    FP = fp(y_true, y_pred)
    FN = fn(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def f_score(y_true, y_pred):
    """2 (precision * recall) / (precision + recall)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec)


def pixel_error(y_true, y_pred):
    pass


def rand_error(y_true, y_pred):
    pass


def wrapping_error(y_true, y_pred):
    pass


def dice(y_true, y_pred):
    """
    DICE = 2|target and prediction| / |target| + |prediction|
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    target = K.sqrt(K.sum(K.square(y_true)))
    prediction = K.sqrt(K.sum(K.square(y_pred)))
    return 2.0 * intersection / (target + prediction)


def iou(y_true, y_pred):
    """
    Intersection over Union (IoU) metric, also known as the Jaccard index.
    Defined as intersection divided by union of the sample sets.

    IoU = |target and prediction| / |target or prediction|
        = |target| + |prediction| - |target or prediction| / |target or prediction|
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return intersection / union
