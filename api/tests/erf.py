from common_import import *
@benchmark_registry .register ("erf")
class ErfConfig (APIConfig ):
    def __init__ (self ):
        super (ErfConfig ,self ).__init__ ("erf")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="activation"
        self .run_torch =False
@benchmark_registry .register ("erf")
class PaddleErf (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .erf (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])