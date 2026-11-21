from common_import import *
@benchmark_registry .register ("unique")
class UniqueConfig (APIConfig ):
    def __init__ (self ):
        super (UniqueConfig ,self ).__init__ ("unique")
        self .run_tf =False
        self .run_torch =False
@benchmark_registry .register ("unique")
class PaddleUnique (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .unique (
        x =x ,
        return_index =config .return_index ,
        return_inverse =config .return_inverse ,
        return_counts =config .return_counts ,
        axis =config .axis ,
        dtype =config .dtype )
        if isinstance (result ,tuple ):
            result =list (result )
        else :
            result =[result ]
        self .feed_list =[x ]
        self .fetch_list =result
@benchmark_registry .register ("unique")
class TorchUnique (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .unique (
        input =x ,
        sorted =True ,
        return_inverse =config .return_inverse ,
        return_counts =config .return_counts ,
        dim =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]