from common_import import *
@benchmark_registry .register ("poisson")
class PoissonConfig (APIConfig ):
    def __init__ (self ):
        super (PoissonConfig ,self ).__init__ ('poisson')
        self .feed_spec =[{"range":[0 ,100 ]}]
@benchmark_registry .register ("poisson")
class PaddlePoisson (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .poisson (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("poisson")
class TorchPoisson (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .poisson (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])