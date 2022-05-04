import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

import parallel_sort

sys.path.append(os.path.abspath('../'))
import tokenizer
#from myutils import index2word

def custom_dist_cce_loss(factor):
    def ls_dist_cce_loss(y_true, y_pred):
        y_true = smooth_labels(y_true, factor)
        cce = keras.losses.categorical_crossentropy(y_true, y_pred)
        return cce
    return ls_dist_cce_loss


def smooth_labels(labels, factor=0.0):
	# smooth the labels
	labels *= (1 - factor)
	labels += (factor / labels.shape[1])

	# returned the smoothed labels
	return labels