from common_import import *
@benchmark_registry .register ("zeros_like")
class PaddleZerosLike (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .zeros_like (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("zeros_like")
class TorchZerosLike (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .zeros_like (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("zeros_like")
class TFZerosLike (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .zeros_like (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]