import numpy as np
import math
import tensorflow as tf
from tensorflow .contrib .rnn import GRUCell
from tensorflow .contrib .rnn import LSTMCell
from tensorflow .python .ops import array_ops
from tensorflow .python .ops .rnn import dynamic_rnn as rnn
from tensorflow .python .ops .rnn import bidirectional_dynamic_rnn as bi_rnn
class PairwiseHingeLoss (object ):
    def __init__ (self ,config ):
        self .margin =float (config ["margin"])
    def ops (self ,score_pos ,score_neg ):
        return tf .reduce_mean (tf .maximum (0. ,score_neg +
        self .margin -score_pos ))
class PairwiseLogLoss (object ):
    def __init__ (self ,config =None ):
        pass
    def ops (self ,score_pos ,score_neg ):
        return tf .reduce_mean (tf .nn .sigmoid (score_neg -score_pos ))
class SoftmaxWithLoss (object ):
    def __init__ (self ):
        pass
    def ops (self ,pred ,label ):
        return tf .reduce_mean (tf .nn .softmax_cross_entropy_with_logits (logits =pred ,
        labels =label ))