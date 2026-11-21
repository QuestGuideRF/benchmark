from common_import import *
@benchmark_registry .register ("diag")
class DiagConfig (APIConfig ):
    def __init__ (self ):
        super (DiagConfig ,self ).__init__ ("diag")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (DiagConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .padding_value !=0 :
            self .run_torch =False
@benchmark_registry .register ("diag")
class PaddleDiag (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .diag (
        x =x ,offset =config .offset ,padding_value =config .padding_value )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("diag")
class TorchDiag (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .diag (input =x ,diagonal =config .offset )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("diag")
class TFDiag (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .linalg .diag (
        diagonal =x ,k =config .offset ,padding_value =config .padding_value )
        self .feed_list =[x ]
        self .fetch_list =[result ]