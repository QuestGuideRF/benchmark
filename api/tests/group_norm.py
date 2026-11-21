from common_import import *
@benchmark_registry .register ("group_norm")
class GroupNormConfig (APIConfig ):
    def __init__ (self ):
        super (GroupNormConfig ,self ).__init__ ('group_norm')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (GroupNormConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .x_shape )==4 :
            self .num_channels =self .x_shape [1 ]if self .data_format =="NCHW"else self .x_shape [3 ]
        else :
            self .num_channels =self .x_shape [1 ]
        if self .data_format =='NHWC':
            print (
            "Warning:\n"
            "1. PyTorch не имеет параметра data_format, он поддерживает только формат NCHW.\n"
            )
            self .run_torch =False
@benchmark_registry .register ("group_norm")
class PaddleGroupNorm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        paddle_group_norm =paddle .nn .GroupNorm (
        num_channels =config .num_channels ,
        num_groups =config .num_groups ,
        epsilon =config .epsilon )
        result =paddle_group_norm (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("group_norm")
class TorchGroupNorm (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        pytorch_group_norm =torch .nn .GroupNorm (
        num_groups =config .num_groups ,
        num_channels =config .num_channels ,
        device =self ._device ,
        eps =config .epsilon )
        result =pytorch_group_norm (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])