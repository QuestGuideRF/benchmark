from common_import import *
@benchmark_registry .register ("batch_norm")
class BatchNormConfig (APIConfig ):
    def __init__ (self ):
        super (BatchNormConfig ,self ).__init__ ('batch_norm')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (BatchNormConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .x_shape )==4 :
            self .num_channels =self .x_shape [
            1 ]if self .data_format =="NCHW"else self .x_shape [3 ]
        else :
            self .num_channels =self .x_shape [1 ]
        self ._set_param_dtype ()
        if self .data_format =='NHWC':
            print (
            "Warning:\n"
            "  1. PyTorch не имеет параметра data_format, он поддерживает только формат NHWC.\n"
            )
            self .run_torch =False
    def _set_param_dtype (self ):
        self .param_dtype ="float32"if self .x_dtype =="float16"else self .x_dtype
    def convert_to_fp16 (self ):
        super (BatchNormConfig ,self ).convert_to_fp16 ()
        if self .data_format =="NHWC":
            paddle .set_flags ({'FLAGS_cudnn_batchnorm_spatial_persistent':1 })
        self ._set_param_dtype ()
@benchmark_registry .register ("batch_norm")
class PaddleBatchNorm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =[config .num_channels ],
        dtype =config .param_dtype )
        bias =self .variable (
        name ='bias',shape =[config .num_channels ],dtype =config .param_dtype )
        if not config .training or not hasattr (self ,"_running_mean"):
            self ._running_mean =paddle .create_parameter (
            name ='running_mean',
            shape =[config .num_channels ],
            dtype =config .param_dtype ,
            attr =paddle .ParamAttr (
            initializer =paddle .nn .initializer .Constant (0 )))
            self ._running_mean .stop_gradient =True
        running_mean =self ._running_mean
        if not config .training or not hasattr (self ,"_running_var"):
            self ._running_var =paddle .create_parameter (
            name ='running_var',
            shape =[config .num_channels ],
            dtype =config .param_dtype ,
            attr =paddle .ParamAttr (
            initializer =paddle .nn .initializer .Constant (1 )))
            self ._running_var .stop_gradient =True
        running_var =self ._running_var
        result =paddle .nn .functional .batch_norm (
        x =x ,
        running_mean =running_mean ,
        running_var =running_var ,
        weight =weight ,
        bias =bias ,
        epsilon =config .epsilon ,
        momentum =config .momentum ,
        training =config .training ,
        data_format =config .data_format )
        self .feed_list =[x ,weight ,bias ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ,bias ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        c =config .num_channels
        flop_mean =numel (x_shape )+c
        flop_var =numel (
        x_shape )*2 +3 *c
        flop_running_mean =3 *c
        flop_running_var =3 *c
        flop_y =numel (
        x_shape )*5
        forward_flop =flop_mean +flop_var +flop_running_mean +flop_running_var +flop_y
        byte_read =numel (x_shape )*sizeof (config .x_dtype )+4 *c *sizeof (
        config .param_dtype )
        byte_write =numel (x_shape )*sizeof (config .x_dtype )+4 *c *sizeof (
        config .param_dtype )
        forward_byte =byte_read +byte_write
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("batch_norm")
class TorchBatchNorm (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =[config .num_channels ],
        dtype =config .param_dtype )
        bias =self .variable (
        name ='bias',shape =[config .num_channels ],dtype =config .param_dtype )
        device =torch .device ("cuda"if use_gpu ()and torch .cuda .is_available ()
        else "cpu")
        if not config .training or not hasattr (self ,"_running_mean"):
            self ._running_mean =torch .zeros (
            [config .num_channels ],device =device )
        running_mean =self ._running_mean
        if not config .training or not hasattr (self ,"_running_var"):
            self ._running_var =torch .ones (
            [config .num_channels ],device =device )
        running_var =self ._running_var
        result =torch .nn .functional .batch_norm (
        input =x ,
        running_mean =running_mean ,
        running_var =running_var ,
        weight =weight ,
        bias =bias ,
        training =config .training ,
        momentum =config .momentum ,
        eps =config .epsilon )
        self .feed_list =[x ,weight ,bias ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ,bias ])