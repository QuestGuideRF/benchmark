from common_import import *
@benchmark_registry .register ("rsqrt")
class RsqrtConfig (APIConfig ):
    def __init__ (self ):
        super (RsqrtConfig ,self ).__init__ ("rsqrt")
        self .feed_spec ={"range":[0 ,10 ]}
        self .alias_name ="activation"
@benchmark_registry .register ("rsqrt")
class PaddleRsqrt (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .rsqrt (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("rsqrt")
class TorchRsqrt (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .rsqrt (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("rsqrt")
class TFRsqrt (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .rsqrt (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])