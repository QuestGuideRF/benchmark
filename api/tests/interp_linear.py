from common_import import *
@benchmark_registry .register ("interp_linear")
class InterpLinearConfig (APIConfig ):
    def __init__ (self ):
        super (InterpLinearConfig ,self ).__init__ ("interp_linear")
        self .run_tf =False
@benchmark_registry .register ("interp_linear")
class PaddleInterpLinear (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="linear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
@benchmark_registry .register ("interp_linear")
class TorchInterpLinear (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="linear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])