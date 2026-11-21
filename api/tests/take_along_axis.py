from common_import import *
@benchmark_registry .register ("take_along_axis")
class TakeAlongAxisConfig (APIConfig ):
    def __init__ (self ):
        super (TakeAlongAxisConfig ,self ).__init__ ("take_along_axis")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[0 ,100 ]}]
        self .run_torch =False
@benchmark_registry .register ("take_along_axis")
class PaddleTakeAlongAxis (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        arr =self .variable (
        name ="arr",shape =config .arr_shape ,dtype =config .arr_dtype )
        indices =self .variable (
        name ="indices",
        shape =config .indices_shape ,
        dtype =config .indices_dtype )
        result =paddle .take_along_axis (
        arr =arr ,indices =indices ,axis =config .axis )
        self .feed_list =[arr ,indices ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[arr ])