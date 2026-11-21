from common_import import *
@benchmark_registry .register ("p_norm")
class PnormConfig (APIConfig ):
    def __init__ (self ):
        super (PnormConfig ,self ).__init__ ("p_norm")
        self .feed_spec =[{"range":[-1 ,1 ]}]
@benchmark_registry .register ("p_norm")
class PaddlePnorm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .norm (
        x =x ,p =config .porder ,axis =config .axis ,keepdim =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("p_norm")
class TorchPnorm (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .norm (
        input =x ,p =config .porder ,dim =config .axis ,keepdim =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("p_norm")
class TFPnorm (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .norm (
        tensor =x ,
        ord =config .porder ,
        axis =config .axis ,
        keepdims =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])