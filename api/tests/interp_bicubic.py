from common_import import *
@benchmark_registry .register ("interp_bicubic")
class InterpBicubicConfig (APIConfig ):
    def __init__ (self ):
        super (InterpBicubicConfig ,self ).__init__ ('interp_bicubic')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (InterpBicubicConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .data_format =='NHWC':
            self .run_torch =False
@benchmark_registry .register ("interp_bicubic")
class PaddleInterpBicubic (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="bilinear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
@benchmark_registry .register ("interp_bicubic")
class TorchInterpBicubic (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="bilinear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])