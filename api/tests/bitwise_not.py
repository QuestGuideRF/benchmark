from common_import import *
@benchmark_registry .register ("bitwise_not")
class BitwiseNotConfig (APIConfig ):
    def __init__ (self ):
        super (BitwiseNotConfig ,self ).__init__ ('bitwise_not')
        self .feed_spec =[{"range":[-100 ,100 ]}]
        self .alias_name ="bitwise"
@benchmark_registry .register ("bitwise_not")
class PaddleBitwiseNot (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .bitwise_not (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("bitwise_not")
class TorchBitwiseNot (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .bitwise_not (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]