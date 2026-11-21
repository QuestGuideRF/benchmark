from common_import import *
@benchmark_registry .register ("isfinite_nan_inf")
class IsfiniteNanInfConfig (APIConfig ):
    def __init__ (self ):
        super (IsfiniteNanInfConfig ,self ).__init__ ("isfinite_nan_inf")
        self .api_name ='isfinite'
        self .api_list ={
        'isfinite':'isfinite',
        'isnan':'isnan',
        'isinf':'isinf'
        }
    def to_tensorflow (self ):
        self .api_list ={
        'isfinite':'is_finite',
        'isnan':'is_nan',
        'isinf':'is_inf'
        }
        tf_config =super (IsfiniteNanInfConfig ,self ).to_tensorflow ()
        return tf_config
@benchmark_registry .register ("isfinite_nan_inf")
class PaddleIsfiniteNanInf (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("isfinite_nan_inf")
class TorchIsfiniteNanInf (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("isfinite_nan_inf")
class TFIsfiniteNanInf (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[out ]