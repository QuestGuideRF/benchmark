from common_import import *
@benchmark_registry .register ("interp_area")
class InterpAreaConfig (APIConfig ):
    def __init__ (self ):
        super (InterpAreaConfig ,self ).__init__ ('interp_area')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (InterpAreaConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .data_format =='NHWC'or self .data_format =='NDHWC':
            self .run_torch =False
@benchmark_registry .register ("interp_area")
class PaddleInterpArea (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="area",
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("interp_area")
class TorchInterpArea (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .interpolate (
        x ,size =config .size ,mode ="area",scale_factor =config .scale_factor )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])