from common_import import *
@benchmark_registry .register ("prelu")
class PReluConfig (APIConfig ):
    def __init__ (self ):
        super (PReluConfig ,self ).__init__ ("prelu")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("prelu")
class PaddlePRelu (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =paddle .nn .functional .prelu (x =x ,weight =weight )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
@benchmark_registry .register ("prelu")
class TorchPRelu (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =torch .nn .functional .prelu (input =x ,weight =weight )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])