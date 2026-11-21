from common_import import *
from conv2d import Conv2dConfig ,PaddleConv2d ,TorchConv2d
@benchmark_registry .register ("depthwise_conv2d",reuse ="conv2d")
class DepthwiseConv2dConfig (Conv2dConfig ):
    def __init__ (self ):
        super (DepthwiseConv2dConfig ,self ).__init__ ("depthwise_conv2d")
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (DepthwiseConv2dConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        assert self .get_in_channels ()==self .groups and self .get_out_channels (
        )%self .get_in_channels ()==0