from common_import import *
@benchmark_registry .register ("interp_bilinear")
class InterpBilinearConfig (APIConfig ):
    def __init__ (self ):
        super (InterpBilinearConfig ,self ).__init__ ('interp_bilinear')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (InterpBilinearConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .data_format =='NHWC':
            print (
            "Warning:\n"
            "  1. PyTorch не имеет параметра data_format, он поддерживает только формат NCHW.\n"
            )
            self .run_torch =False
@benchmark_registry .register ("interp_bilinear")
class PaddleInterpBilinear (PaddleOpBenchmarkBase ):
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
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        out_shape =self .fetch_list [0 ].shape
        forward_flop =numel (out_shape )*23
        forward_byte =(
        numel (x_shape )+numel (out_shape ))*sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("interp_bilinear")
class TorchInterpBilinear (PytorchOpBenchmarkBase ):
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