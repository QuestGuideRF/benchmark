from common_import import *
@benchmark_registry .register ("triu")
class TriuConfig (APIConfig ):
    def __init__ (self ):
        super (TriuConfig ,self ).__init__ ("triu")
        self .run_tf =False
@benchmark_registry .register ("triu")
class PaddleTriu (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .triu (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("triu")
class TorchTriu (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .triu (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])