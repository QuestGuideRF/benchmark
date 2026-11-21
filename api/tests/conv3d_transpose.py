from common_import import *
@benchmark_registry .register ("conv3d_transpose")
class Conv3dTransposeConfig (APIConfig ):
    def __init__ (self ,op_type ="conv3d_transpose"):
        super (Conv3dTransposeConfig ,self ).__init__ (op_type )
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        }
        ]
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Conv3dTransposeConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if isinstance (self .padding ,int ):
            self .padding =[self .padding ,self .padding ,self .padding ]
        if self .data_format =="NCDHW":
            self .num_channels =self .input_shape [1 ]
        elif self .data_format =="NDHWC":
            self .num_channels =self .input_shape [4 ]
        if isinstance (self .filter_size ,int ):
            self .filter_size =[
            self .filter_size ,self .filter_size ,self .filter_size
            ]
        if self .groups is None :
            self .groups =1
        if self .num_channels %self .groups !=0 :
            raise ValueError (
            "канал ввода должен делиться на группы,"
            "получено: канал ввода — {}, форма ввода — {}"
            ", группы {}".format (self .num_channels ,
            self .input_shape ,self .groups ))
        self .filter_shape =[
        self .num_channels //self .groups ,self .num_filters ,
        self .filter_size [0 ],self .filter_size [1 ],self .filter_size [2 ]
        ]
@benchmark_registry .register ("conv3d_transpose")
class PaddleConv3dTranspose (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        weight =self .variable (
        name ='weight',shape =config .filter_shape ,dtype =config .input_dtype )
        result =paddle .nn .functional .conv3d_transpose (
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
@benchmark_registry .register ("conv3d_transpose")
class TorchConv3dTranspose (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        weight =self .variable (
        name ='weight',shape =config .filter_shape ,dtype =config .input_dtype )
        result =torch .nn .functional .conv_transpose3d (
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