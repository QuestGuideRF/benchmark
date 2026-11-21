from common_import import *
@benchmark_registry .register ("arg")
class ArgConfig (APIConfig ):
    def __init__ (self ):
        super (ArgConfig ,self ).__init__ ('arg')
        self .api_name ='argmax'
        self .api_list ={'argmax':'argmax','argmin':'argmin'}
@benchmark_registry .register ("arg")
class PaddleArg (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("arg")
class TorchArg (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,input =x ,dim =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("arg")
class TFArg (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,input =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]