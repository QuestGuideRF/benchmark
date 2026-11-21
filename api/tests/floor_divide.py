from common_import import *
@benchmark_registry .register ("floor_divide")
class FloorDivideConfig (APIConfig ):
    def __init__ (self ):
        super (FloorDivideConfig ,self ).__init__ ("floor_divide")
        self .feed_spec =[{"range":[1 ,1000 ]},{"range":[1 ,1000 ]}]
@benchmark_registry .register ("floor_divide")
class PaddleFloorDivide (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .floor_divide (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
@benchmark_registry .register ("floor_divide")
class TorchFloorDivide (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =torch .floor_divide (input =x ,other =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
@benchmark_registry .register ("floor_divide")
class TFFloorDivide (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =tf .math .floordiv (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]