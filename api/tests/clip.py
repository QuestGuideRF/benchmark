from common_import import *
@benchmark_registry .register ("clip")
class ClipConfig (APIConfig ):
    def __init__ (self ):
        super (ClipConfig ,self ).__init__ ("clip")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .run_torch =False
@benchmark_registry .register ("clip")
class PaddleClip (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .clip (x =x ,min =config .min ,max =config .max )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])