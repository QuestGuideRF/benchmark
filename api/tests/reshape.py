from common_import import *
@benchmark_registry .register ("reshape")
class PaddleReshape (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .reshape (x =x ,shape =config .shape )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("reshape")
class TorchReshape (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .reshape (x =x ,shape =config .shape )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("reshape")
class TFReshape (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .reshape (tensor =data ,shape =config .shape )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])