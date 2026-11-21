from common_import import *
@benchmark_registry .register ("affine_grid")
class AffineGridConfig (APIConfig ):
    def __init__ (self ):
        super (AffineGridConfig ,self ).__init__ ("affine_grid")
        self .run_tf =False
@benchmark_registry .register ("affine_grid")
class PaddleAffineGrid (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        theta =self .variable (
        name ='theta',shape =config .theta_shape ,dtype =config .theta_dtype )
        result =paddle .nn .functional .affine_grid (
        theta ,
        out_shape =config .out_shape ,
        align_corners =config .align_corners )
        self .feed_list =[theta ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[theta ])
@benchmark_registry .register ("affine_grid")
class TorchAffineGrid (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        theta =self .variable (
        name ='theta',shape =config .theta_shape ,dtype =config .theta_dtype )
        result =torch .nn .functional .affine_grid (
        theta ,size =config .out_shape ,align_corners =config .align_corners )
        self .feed_list =[theta ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[theta ])