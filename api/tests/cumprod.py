from common_import import *
@benchmark_registry .register ("cumprod")
class CumprodConfig (APIConfig ):
    def __init__ (self ):
        super (CumprodConfig ,self ).__init__ ('cumprod')
        self .feed_spec ={"range":[-1 ,1 ]}
@benchmark_registry .register ("cumprod")
class PaddleCumprod (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .cumprod (x =x ,dim =config .dim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("cumprod")
class TorchCumprod (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .cumprod (x =x ,dim =config .dim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])