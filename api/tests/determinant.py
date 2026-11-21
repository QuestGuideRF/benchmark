from common_import import *
@benchmark_registry .register ("determinant")
class DeterminantConfig (APIConfig ):
    def __init__ (self ):
        super (DeterminantConfig ,self ).__init__ ('determinant')
        self .feed_spec =[{"range":[-1 ,1 ]}]
@benchmark_registry .register ("determinant")
class PaddleDeterminant (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .linalg .det (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])