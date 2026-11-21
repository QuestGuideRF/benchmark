from common_import import *
@benchmark_registry .register ("diagonal")
class PaddleDiagonal (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .diagonal (
        x =x ,offset =config .offset ,axis1 =config .axis1 ,axis2 =config .axis2 )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("diagonal")
class TorchDiagonal (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .diagonal (
        input =x ,
        offset =config .offset ,
        dim1 =config .axis1 ,
        dim2 =config .axis2 ).contiguous ()
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])