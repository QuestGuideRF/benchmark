from common_import import *
@benchmark_registry .register ("linear")
class LinearConfig (APIConfig ):
    def __init__ (self ):
        super (LinearConfig ,self ).__init__ ('linear')
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        "permute":[1 ,0 ]
        },
        {
        "range":[-1 ,1 ]
        }
        ]
    def to_pytorch (self ):
        torch_config =super (LinearConfig ,self ).to_pytorch ()
        torch_config .weight_shape =[
        self .weight_shape [1 ],self .weight_shape [0 ]
        ]
        return torch_config
@benchmark_registry .register ("linear")
class PaddleLinear (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ="weight",
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        if hasattr (config ,"bias")and config .bias is None :
            bias =None
        else :
            bias =self .variable (
            name ="bias",shape =config .bias_shape ,dtype =config .bias_dtype )
        result =paddle .nn .functional .linear (x =x ,weight =weight ,bias =bias )
        self .feed_list =[x ,weight ,bias ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ,bias ])
@benchmark_registry .register ("linear")
class TorchLinear (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ="weight",
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        if hasattr (config ,"bias")and config .bias is None :
            bias =None
        else :
            bias =self .variable (
            name ="bias",shape =config .bias_shape ,dtype =config .bias_dtype )
        result =torch .nn .functional .linear (input =x ,weight =weight ,bias =bias )
        self .feed_list =[x ,weight ,bias ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ,bias ])