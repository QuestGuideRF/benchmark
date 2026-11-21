from common_import import *
@benchmark_registry .register ("bincount")
class PaddleBincount (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        w =self .variable (name ='w',shape =config .w_shape ,dtype =config .w_dtype )
        result =paddle .bincount (x =x ,weights =w ,minlength =config .minlength )
        self .feed_list =[x ,w ]
        self .fetch_list =[result ]
@benchmark_registry .register ("bincount")
class TorchBincount (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        w =self .variable (name ='w',shape =config .w_shape ,dtype =config .w_dtype )
        result =torch .bincount (input =x ,weights =w ,minlength =config .minlength )
        self .feed_list =[x ,w ]
        self .fetch_list =[result ]