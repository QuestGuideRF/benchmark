from common_import import *
@benchmark_registry .register ("bitwise")
class BitwiseConfig (APIConfig ):
    def __init__ (self ):
        super (BitwiseConfig ,self ).__init__ ('bitwise')
        self .api_name ='bitwise_and'
        self .api_list ={
        'bitwise_and':'bitwise_and',
        'bitwise_or':'bitwise_or',
        'bitwise_xor':'bitwise_xor'
        }
        self .feed_spec =[{"range":[-100 ,100 ]},{"range":[-100 ,100 ]}]
@benchmark_registry .register ("bitwise")
class PaddleBitwise (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
@benchmark_registry .register ("bitwise")
class TorchBitwise (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,input =x ,other =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]