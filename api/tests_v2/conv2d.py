from common_import import *
class Conv2dConfig (APIConfig ):
    def __init__ (self ,op_type ="conv2d"):
        super (Conv2dConfig ,self ).__init__ (op_type )
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        "permute":[2 ,3 ,1 ,0 ]
        }
        ]
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Conv2dConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if not use_gpu ()and self .data_format =="NCHW":
            print (
            "Warning:\n"
            "  1. tf отключен, потому что в настоящее время используются только конверсии tf. "
            "поддерживает тензорный формат NHWC на ​​ЦП. Пожалуйста, добавьте правило "
            "чтобы поддержать его.\n")
            self .run_tf =False
        if isinstance (self .padding ,int ):
            self .padding =[self .padding ,self .padding ]
        if self .data_format =="NCHW":
            self .num_channels =self .x_shape [1 ]
        elif self .data_format =="NHWC":
            self .num_channels =self .x_shape [3 ]
        if self .groups is None :
            self .groups =1
        if self .num_channels %self .groups !=0 :
            raise ValueError (
            "канал ввода должен делиться на группы,"
            "получено: канал ввода — {}, форма ввода — {}"
            ", группы {}".format (self .num_channels ,self .x_shape ,
            self .groups ))
    def to_tensorflow (self ):
        tf_config =super (Conv2dConfig ,self ).to_tensorflow ()
        tf_config .weight_shape =[
        self .weight_shape [2 ],self .weight_shape [3 ],self .weight_shape [1 ],
        self .weight_shape [0 ]
        ]
        tf_config .padding =self ._convert_padding (self .padding )
        return tf_config
    def _convert_padding (self ,padding ):
        if isinstance (padding ,str ):
            return padding
        assert isinstance (padding ,list )
        assert len (padding )==2 or len (padding )==4
        pad_top =padding [0 ]if len (padding )==2 else padding [0 ]
        pad_bottom =padding [0 ]if len (padding )==2 else padding [1 ]
        pad_left =padding [1 ]if len (padding )==2 else padding [2 ]
        pad_right =padding [1 ]if len (padding )==2 else padding [3 ]
        if self .data_format =="NCHW":
            return [[0 ,0 ],[0 ,0 ],[pad_top ,pad_bottom ],
            [pad_left ,pad_right ]]
        elif self .data_format =="NHWC":
            return [[0 ,0 ],[pad_top ,pad_bottom ],[pad_left ,pad_right ],
            [0 ,0 ]]
class PDConv2d (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
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
        self .feed_vars =[x ,weight ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
    def compute_flop_and_byte (self ,config ):
        filter_shape =self .feed_vars [1 ].shape
        output_shape =self .fetch_vars [0 ].shape
        if config .data_format =="NCHW":
            M =output_shape [0 ]*output_shape [2 ]*output_shape [3 ]
            N =output_shape [1 ]
        elif data_format =="NHWC":
            M =output_shape [0 ]*output_shape [1 ]*output_shape [2 ]
            N =output_shape [3 ]
        K =filter_shape [1 ]*filter_shape [2 ]*filter_shape [3 ]
        forward_flop =2 *M *N *K
        print ("Вперед: M = %d, N = %d, K = %d, gflop = %.5f"%
        (M ,N ,K ,float (forward_flop )*1E-9 ))
        if not config .backward :
            return forward_flop ,None
        else :
            return None ,None
class TFConv2d (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =tf .nn .conv2d (
        input =x ,
        filters =weight ,
        strides =config .stride ,
        padding =config .padding ,
        data_format =config .data_format ,
        dilations =config .dilation )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
if __name__ =='__main__':
    test_main (PDConv2d (),TFConv2d (),config =Conv2dConfig ())