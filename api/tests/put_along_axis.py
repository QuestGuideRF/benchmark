from common_import import *
@benchmark_registry .register ("put_along_axis")
class PutAlongAxisConfig (APIConfig ):
    def __init__ (self ):
        super (PutAlongAxisConfig ,self ).__init__ ("put_along_axis")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[0 ,100 ]}]
        self .run_torch =False
@benchmark_registry .register ("put_along_axis")
class PaddlePutAlongAxis (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        arr =self .variable (
        name ="arr",shape =config .arr_shape ,dtype =config .arr_dtype )
        indices =self .variable (
        name ="indices",
        shape =config .indices_shape ,
        dtype =config .indices_dtype )
        result =paddle .put_along_axis (
        arr =arr ,indices =indices ,values =config .values ,axis =config .axis )
        self .feed_list =[arr ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[arr ])