from common_import import *
@benchmark_registry .register ("bmm")
class BmmConfig (APIConfig ):
    def __init__ (self ):
        super (BmmConfig ,self ).__init__ ("bmm")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
        self .run_tf =False
@benchmark_registry .register ("bmm")
class PaddleBmm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .bmm (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
@benchmark_registry .register ("bmm")
class TorchBmm (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =torch .bmm (input =x ,mat2 =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])