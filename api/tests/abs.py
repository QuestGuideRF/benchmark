from common_import import *
@benchmark_registry .register ("abs")
class AbsConfig (APIConfig ):
    def __init__ (self ):
        super (AbsConfig ,self ).__init__ ("abs")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="activation"
@benchmark_registry .register ("abs")
class PaddleAbs (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .abs (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        forward_flop =numel (x_shape )
        forward_byte =numel (x_shape )*2 *sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("abs")
class TorchAbs (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .abs (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("abs")
class TFAbs (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .abs (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])