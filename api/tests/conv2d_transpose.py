from common_import import *
@benchmark_registry .register ("conv2d_transpose")
class Conv2dTransposeConfig (APIConfig ):
    def __init__ (self ):
        super (Conv2dTransposeConfig ,self ).__init__ ("conv2d_transpose")
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        }
        ]
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Conv2dTransposeConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .groups ==None :
            self .groups =1
@benchmark_registry .register ("conv2d_transpose")
class PaddleConv2dTranspose (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ="weight",
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =paddle .nn .functional .conv2d_transpose (
        x =x ,
        weight =weight ,
        bias =None ,
        stride =config .stride ,
        padding =config .padding ,
        output_padding =0 ,
        dilation =config .dilation ,
        groups =config .groups ,
        output_size =config .output_size ,
        data_format =config .data_format )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
@benchmark_registry .register ("conv2d_transpose")
class TorchConv2dTranspose (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ="weight",
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =torch .nn .functional .conv_transpose2d (
        input =x ,
        weight =weight ,
        bias =None ,
        stride =config .stride ,
        padding =config .padding ,
        output_padding =0 ,
        dilation =config .dilation ,
        groups =config .groups )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])