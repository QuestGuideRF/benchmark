from common_import import *
@benchmark_registry .register ("binary_cross_entropy_with_logits")
class BinaryCrossEntropyWithLogitsConfig (APIConfig ):
    def __init__ (self ):
        super (BinaryCrossEntropyWithLogitsConfig ,
        self ).__init__ ("binary_cross_entropy_with_logits")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (BinaryCrossEntropyWithLogitsConfig ,self ).init_from_json (
        filename ,config_id ,unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,self .logit_shape [-1 ]]
        }
        ]
@benchmark_registry .register ("binary_cross_entropy_with_logits")
class PaddleBinaryCrossEntropyWithLogits (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        logit =self .variable (
        name ="logit",shape =config .logit_shape ,dtype =config .logit_dtype )
        label =self .variable (
        name ="label",
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        stop_gradient =True )
        result =paddle .nn .functional .binary_cross_entropy_with_logits (
        logit =logit ,
        label =label ,
        weight =None ,
        reduction ="none",
        pos_weight =None )
        self .feed_list =[logit ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[logit ])
@benchmark_registry .register ("binary_cross_entropy_with_logits")
class TorchBinaryCrossEntropyWithLogits (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        logit =self .variable (
        name ="logit",shape =config .logit_shape ,dtype =config .logit_dtype )
        label =self .variable (
        name ="label",
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        stop_gradient =True )
        result =torch .nn .functional .binary_cross_entropy_with_logits (
        input =logit ,target =label ,weight =None ,reduction ="none")
        self .feed_list =[logit ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[logit ])
@benchmark_registry .register ("binary_cross_entropy_with_logits")
class TFBinaryCrossEntropyWithLogits (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        logit =self .variable (
        name ='logit',shape =config .logit_shape ,dtype =config .logit_dtype )
        label =self .variable (
        name ='label',shape =config .label_shape ,dtype =config .label_dtype )
        result =tf .nn .sigmoid_cross_entropy_with_logits (
        logits =logit ,labels =label )
        self .feed_list =[logit ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[logit ])