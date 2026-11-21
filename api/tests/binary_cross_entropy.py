from common_import import *
@benchmark_registry .register ("binary_cross_entropy")
class BinaryCrossEntropyConfig (APIConfig ):
    def __init__ (self ):
        super (BinaryCrossEntropyConfig ,self ).__init__ ("binary_cross_entropy")
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (BinaryCrossEntropyConfig ,self ).init_from_json (
        filename ,config_id ,unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,self .input_shape [-1 ]]
        }
        ]
@benchmark_registry .register ("binary_cross_entropy")
class PaddleBinaryCrossEntropy (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ="input",shape =config .input_shape ,dtype =config .input_dtype )
        label =self .variable (
        name ="label",
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        stop_gradient =True )
        result =paddle .nn .functional .binary_cross_entropy (
        input =input ,label =label ,weight =None ,reduction ="none")
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
@benchmark_registry .register ("binary_cross_entropy")
class TorchBinaryCrossEntropy (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ="input",shape =config .input_shape ,dtype =config .input_dtype )
        label =self .variable (
        name ="label",
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        stop_gradient =True )
        result =torch .nn .functional .binary_cross_entropy (
        input =input ,target =label ,weight =None ,reduction ="none")
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])