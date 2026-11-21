from common_import import *
@benchmark_registry .register ("gelu")
class GeluConfig (APIConfig ):
    def __init__ (self ):
        super (GeluConfig ,self ).__init__ ("gelu")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .run_tf =False
@benchmark_registry .register ("gelu")
class PaddleGelu (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .gelu (x =x ,approximate =config .approximate )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("gelu")
class TorchGelu (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .gelu (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])