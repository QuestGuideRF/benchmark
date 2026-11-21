from common_import import *
@benchmark_registry .register ("size")
class PaddleSize (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .numel (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("size")
class TorchSize (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .numel (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]