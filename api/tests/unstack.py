from common_import import *
@benchmark_registry .register ("unstack")
class UnstackConfig (APIConfig ):
    def __init__ (self ):
        super (UnstackConfig ,self ).__init__ ('unstack')
        self .run_torch =False
        print ("[ВНИМАНИЕ]: Pytorch в настоящее время не поддерживает разбиение.")
@benchmark_registry .register ("unstack")
class PaddleUnStack (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .unstack (x =x ,axis =config .axis )
        self .feed_list =[x ]
        if type (result )==list :
            self .fetch_list =result
            config .backward =False
        else :
            self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        forward_flop =0
        forward_byte =numel (config .x_shape )*sizeof (config .x_dtype )*2
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None