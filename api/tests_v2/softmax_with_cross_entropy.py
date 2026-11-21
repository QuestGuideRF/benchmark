from common_import import *
class SoftmaxWithCrossEntropyConfig (APIConfig ):
    def __init__ (self ):
        super (SoftmaxWithCrossEntropyConfig ,
        self ).__init__ ("softmax_with_cross_entropy")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (SoftmaxWithCrossEntropyConfig ,self ).init_from_json (
        filename ,config_id ,unknown_dim )
        if self .axis ==len (self .logits_shape )-1 :
            self .axis =-1
        self .num_classes =self .logits_shape [self .axis ]
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,self .num_classes ]
        }
        ]
        if self .label_dtype in ['float32','float64']or self .axis !=-1 :
            self .run_tf =False
    def to_tensorflow (self ):
        tf_config =super (SoftmaxWithCrossEntropyConfig ,self ).to_tensorflow ()
        label_rank =len (tf_config .label_shape )
        if tf_config .label_shape [label_rank -1 ]==1 :
            tf_config .label_shape =tf_config .label_shape [0 :label_rank -1 ]
        return tf_config
class PDSoftmaxWithCrossEntropy (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        logits =self .variable (
        name ='logits',
        shape =config .logits_shape ,
        dtype =config .logits_dtype )
        label =self .variable (
        name ="label",
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        stop_gradient =True )
        result =paddle .nn .functional .softmax_with_cross_entropy (
        logits =logits ,
        label =label ,
        soft_label =config .soft_label ,
        axis =config .axis )
        self .feed_vars =[logits ,label ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[logits ])
class TFSoftmaxWithCrossEntropy (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        logits =self .variable (
        name ='logits',
        shape =config .logits_shape ,
        dtype =config .logits_dtype )
        label =self .variable (
        name ='label',shape =config .label_shape ,dtype =config .label_dtype )
        onehot_label =tf .one_hot (indices =label ,depth =config .num_classes )
        result =tf .compat .v1 .losses .softmax_cross_entropy (
        logits =logits ,onehot_labels =onehot_label ,reduction ='none')
        self .feed_list =[logits ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[logits ])
if __name__ =='__main__':
    test_main (
    PDSoftmaxWithCrossEntropy (),
    TFSoftmaxWithCrossEntropy (),
    config =SoftmaxWithCrossEntropyConfig ())