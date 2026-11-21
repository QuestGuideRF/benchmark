from common_import import *
@benchmark_registry .register ("tril")
class TrilConfig (APIConfig ):
    def __init__ (self ):
        super (TrilConfig ,self ).__init__ ("tril")
        self .run_tf =False
@benchmark_registry .register ("tril")
class PaddleTril (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .tril (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("tril")
class TorchTril (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .tril (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])