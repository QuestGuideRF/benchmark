from common_import import *
@benchmark_registry .register ("maxout")
class MaxoutConfig (APIConfig ):
    def __init__ (self ):
        super (MaxoutConfig ,self ).__init__ ('maxout')
        self .feed_spec ={"range":[-1 ,1 ]}
        self .run_torch =False
@benchmark_registry .register ("maxout")
class PaddleMaxout (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .maxout (
        x =x ,groups =config .groups ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])