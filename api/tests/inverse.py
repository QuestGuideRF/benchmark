from common_import import *
@benchmark_registry .register ("inverse")
class PaddleInverse (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .inverse (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("inverse")
class TorchInverse (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .inverse (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("inverse")
class TFInverse (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .linalg .inv (input =input )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])