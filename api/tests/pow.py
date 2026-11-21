from common_import import *
from elementwise import ElementwiseConfig
@benchmark_registry .register ("pow")
class PowConfig (ElementwiseConfig ):
    def __init__ (self ):
        super (PowConfig ,self ).__init__ ('pow')
        self .alias_name ='elementwise'
        self .api_name ='pow'
        self .api_list ={'pow':'pow'}
        self .run_tf =False
@benchmark_registry .register ("pow")
class PaddlePow (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .pow (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("pow")
class TorchPow (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =torch .pow (input =x ,exponent =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )