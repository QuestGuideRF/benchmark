import logging
from static_graph .MMDNN .paddle import layers as layers
class KNRM (object ):
    def __init__ (self ,config ):
        self .vocab_size =int (config ['vocabulary_size'])
        self .emb_size =int (config ['embedding_dim'])
        self .kernel_num =int (config ['kernel_num'])
        self .left_name ,self .seq_len1 =config ['left_slots'][0 ]
        self .right_name ,self .seq_len2 =config ['right_slots'][0 ]
        self .lamb =float (config ['lamb'])
        self .task_mode =config ['training_mode']
        self .emb_layer =layers .EmbeddingLayer (self .vocab_size ,self .emb_size )
        self .sim_mat_layer =layers .SimilarityMatrixLayer ()
        self .kernel_pool_layer =layers .KernelPoolingLayer (self .kernel_num ,self .lamb )
        self .tanh_layer =layers .TanhLayer ()
        if self .task_mode =="pointwise":
            self .n_class =int (config ['n_class'])
            self .fc_layer =layers .FCLayer (self .kernel_num ,self .n_class )
        elif self .task_mode =="pairwise":
            self .fc_layer =layers .FCLayer (self .kernel_num ,1 )
        else :
            logging .error ("режим обучения не поддерживается")
    def predict (self ,left_slots ,right_slots ):
        left =left_slots [self .left_name ]
        right =right_slots [self .right_name ]
        left_emb =self .emb_layer .ops (left )
        right_emb =self .emb_layer .ops (right )
        sim_mat =self .sim_mat_layer .ops (left_emb ,right_emb )
        feats =self .kernel_pool_layer .ops (sim_mat )
        pred =self .fc_layer .ops (feats )
        return pred