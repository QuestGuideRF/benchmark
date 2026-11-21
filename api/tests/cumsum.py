from common_import import *
@benchmark_registry .register ("cumsum")
class CumsumConfig (APIConfig ):
    def __init__ (self ):
        super (CumsumConfig ,self ).__init__ ('cumsum')
        self .feed_spec ={"range":[-1 ,1 ]}
@benchmark_registry .register ("cumsum")
class PaddleCumsum (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .cumsum (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("cumsum")
class TorchCumsum (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .cumsum (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("cumsum")
class TFCumsum (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .cumsum (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]