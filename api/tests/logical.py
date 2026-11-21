from common_import import *
@benchmark_registry .register ("logical")
class LogicalConfig (APIConfig ):
    def __init__ (self ):
        super (LogicalConfig ,self ).__init__ ('logical')
        self .api_name ='logical_and'
        self .api_list ={
        'logical_and':'logical_and',
        'logical_or':'logical_or',
        'logical_xor':'logical_xor'
        }
@benchmark_registry .register ("logical")
class PaddleLogical (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
@benchmark_registry .register ("logical")
class TorchLogical (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,input =x ,other =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
@benchmark_registry .register ("logical")
class TFLogical (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]