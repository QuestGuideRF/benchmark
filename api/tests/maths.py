from common_import import *
@benchmark_registry .register ("maths")
class MathsConfig (APIConfig ):
    def __init__ (self ):
        super (MathsConfig ,self ).__init__ ('maths')
        self .api_name ='cos'
        self .api_list ={
        'cos':'cos',
        'exp':'exp',
        'lgamma':'lgamma',
        'log':'log',
        'sin':'sin',
        'sinh':'sinh',
        'sqrt':'sqrt',
        'square':'square',
        'tanh':'tanh'
        }
        self .alias_name ="activation"
@benchmark_registry .register ("maths")
class PaddleMaths (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("maths")
class TorchMaths (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("maths")
class TFActivation (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])