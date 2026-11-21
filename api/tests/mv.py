from common_import import *
@benchmark_registry .register ("mv")
class MvConfig (APIConfig ):
    def __init__ (self ):
        super (MvConfig ,self ).__init__ ("mv")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("mv")
class PaddleMv (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        vec =self .variable (
        name ='vec',shape =config .vec_shape ,dtype =config .vec_dtype )
        result =paddle .mv (x =x ,vec =vec )
        self .feed_list =[x ,vec ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,vec ])
@benchmark_registry .register ("mv")
class TorchMv (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        vec =self .variable (
        name ='vec',shape =config .vec_shape ,dtype =config .vec_dtype )
        result =torch .mv (x =x ,vec =vec )
        self .feed_list =[x ,vec ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,vec ])
@benchmark_registry .register ("mv")
class TFMv (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        vec =self .variable (
        name ='vec',shape =config .vec_shape ,dtype =config .vec_dtype )
        result =tf .linalg .matvec (
        a =x ,
        b =vec ,
        transpose_a =False ,
        adjoint_a =False ,
        a_is_sparse =False ,
        b_is_sparse =False )
        self .feed_list =[x ,vec ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,vec ])