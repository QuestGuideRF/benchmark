from common_import import *
@benchmark_registry .register ("ones_like")
class OnesLikeConfig (APIConfig ):
    def __init__ (self ):
        super (OnesLikeConfig ,self ).__init__ ('ones_like')
        self .alias_name ='zeros_like'
@benchmark_registry .register ("ones_like")
class PaddleOnesLike (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .ones_like (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("ones_like")
class TorchOnesLike (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .ones_like (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("ones_like")
class TorchOnesLike (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .ones_like (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]