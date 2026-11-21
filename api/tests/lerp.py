from common_import import *
@benchmark_registry .register ("lerp")
class PaddleLerp (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .lerp (x =x ,y =y ,weight =config .w )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
@benchmark_registry .register ("lerp")
class TorchLerp (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =torch .lerp (input =x ,end =y ,weight =config .w )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])