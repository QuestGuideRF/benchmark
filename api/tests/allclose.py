from common_import import *
@benchmark_registry .register ("allclose")
class AllcloseConfig (APIConfig ):
    def __init__ (self ):
        super (AllcloseConfig ,self ).__init__ ("allclose")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("allclose")
class PaddleAllclose (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .allclose (
        x =x ,
        y =y ,
        rtol =config .rtol ,
        atol =config .atol ,
        equal_nan =config .equal_nan )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]