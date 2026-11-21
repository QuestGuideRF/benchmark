from common_import import *
@benchmark_registry .register ("conv2d")
class Conv2dConfig (APIConfig ):
    def __init__ (self ,op_type ="conv2d"):
        super (Conv2dConfig ,self ).__init__ (op_type )
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        }
        ]
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Conv2dConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if isinstance (self .padding ,int ):
            self .padding =[self .padding ,self .padding ]
        if isinstance (self .dilation ,int ):
            self .dilation =[self .dilation ,self .dilation ]
        if self .groups is None :
            self .groups =1
        assert self .get_in_channels (
        )%self .groups ==0 ,"Канал ввода должен делиться на группы. "  "Но получено: канал ввода {}, форма ввода {}, группы {}".format (
        self .get_in_channels (),self .x_shape ,self .groups )
        if self .data_format =='NHWC':
            print (
            "Warning:\n"
            "  1. PyTorch не имеет параметра data_format, он поддерживает только формат NCHW.\n"
            )
            self .run_torch =False
    def get_in_channels (self ):
        return self .x_shape [1 ]if self .data_format =="NCHW"else self .x_shape [
        3 ]
    def get_out_channels (self ):
        return self .weight_shape [0 ]
@benchmark_registry .register ("conv2d")
class PaddleConv2d (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =paddle .nn .functional .conv2d (
        x =x ,
        weight =weight ,
        bias =None ,
        stride =config .stride ,
        padding =config .padding ,
        dilation =config .dilation ,
        groups =config .groups ,
        data_format =config .data_format )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
@benchmark_registry .register ("conv2d")
class TorchConv2d (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',shape =config .weight_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .conv2d (
        input =x ,
        weight =weight ,
        bias =None ,
        stride =config .stride ,
        padding =config .padding ,
        dilation =config .dilation ,
        groups =config .groups )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])