from common_import import *
@benchmark_registry .register ("unsqueeze")
class UnsqueezeConfig (APIConfig ):
    def __init__ (self ):
        super (UnsqueezeConfig ,self ).__init__ ("unsqueeze")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (UnsqueezeConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .axis ==[2 ]:
            self .axis =2
@benchmark_registry .register ("unsqueeze")
class PaddleUnsqueeze (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .unsqueeze (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("unsqueeze")
class TorchUnsqueeze (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .unsqueeze (input =x ,dim =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("unsqueeze")
class TFUnsqueeze (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .expand_dims (input =input ,axis =config .axis )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])