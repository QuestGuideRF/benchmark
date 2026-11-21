from common_import import *
@benchmark_registry .register ("dist")
class DistConfig (APIConfig ):
    def __init__ (self ):
        super (DistConfig ,self ).__init__ ("dist")
        self .run_tf =False
@benchmark_registry .register ("dist")
class PaddleDist (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .dist (x =x ,y =y ,p =config .p )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("dist")
class TorchDist (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =torch .dist (input =x ,other =y ,p =config .p )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )