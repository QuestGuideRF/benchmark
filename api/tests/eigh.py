from common_import import *
@benchmark_registry .register ("eigh")
class EighConfig (APIConfig ):
    def __init__ (self ):
        super (EighConfig ,self ).__init__ ("eigh")
        self .feed_spec =[{"range":[-1 ,1 ]}]
@benchmark_registry .register ("eigh")
class PaddleEigh (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out_w ,out_v =paddle .linalg .eigh (x =x ,UPLO =config .UPLO )
        self .feed_list =[x ]
        self .fetch_list =[out_w ,out_v ]
        if config .backward :
            self .append_gradients (out_w .sum ()+paddle .abs (out_v ).sum (),[x ])
@benchmark_registry .register ("eigh")
class TorchEigh (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out_w ,out_v =torch .linalg .eigh (input =x ,UPLO =config .UPLO )
        self .feed_list =[x ]
        self .fetch_list =[out_w ,out_v ]
        if config .backward :
            self .append_gradients (out_w .sum ()+torch .abs (out_v ).sum (),[x ])