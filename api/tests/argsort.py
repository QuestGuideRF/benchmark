from common_import import *
@benchmark_registry .register ("argsort")
class ArgsortConfig (APIConfig ):
    def __init__ (self ):
        super (ArgsortConfig ,self ).__init__ ("argsort")
    def to_tensorflow (self ):
        tf_config =super (ArgsortConfig ,self ).to_tensorflow ()
        if self .descending :
            tf_config .direction ="DESCENDING"
        else :
            tf_config .direction ="ASCENDING"
        return tf_config
@benchmark_registry .register ("argsort")
class PaddleArgsort (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        indices =paddle .argsort (
        x =x ,axis =config .axis ,descending =config .descending )
        self .feed_list =[x ]
        self .fetch_list =[indices ]
@benchmark_registry .register ("argsort")
class TorchArgsort (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        indices =torch .argsort (
        x =x ,axis =config .axis ,descending =config .descending )
        self .feed_list =[x ]
        self .fetch_list =[indices ]
@benchmark_registry .register ("argsort")
class TFArgsort (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        indices =tf .argsort (
        values =x ,
        axis =config .axis ,
        direction =config .direction ,
        stable =False )
        self .feed_list =[x ]
        self .fetch_list =[indices ]