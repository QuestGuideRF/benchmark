from common_import import *
@benchmark_registry .register ("instance_norm")
class InstanceNormConfig (APIConfig ):
    def __init__ (self ):
        super (InstanceNormConfig ,self ).__init__ ('instance_norm')
        self .run_tf =False
@benchmark_registry .register ("instance_norm")
class PaddleInstanceNorm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .instance_norm (x =x ,eps =config .eps )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("instance_norm")
class TorchInstanceNorm (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .instance_norm (input =x ,eps =config .eps )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])