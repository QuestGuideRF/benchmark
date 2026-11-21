from common_import import *
from conv2d import Conv2dConfig
class Conv2dTransposeConfig (Conv2dConfig ):
    def __init__ (self ):
        super (Conv2dTransposeConfig ,self ).__init__ ("conv2d_transpose")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Conv2dTransposeConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .filter_shape =[
        self .num_channels //self .groups ,self .num_filters ,
        self .filter_size [0 ],self .filter_size [1 ]
        ]
        if not isinstance (
        self .padding ,
        str )and self .padding !=[0 ,0 ]and self .padding !=[1 ,1 ]:
            print (
            "Warning:\n"
            "  1. tf отключен, поскольку заполнение аргументов tf "
            "conv2d_transpose должен быть строкой и иметь значение \"SAME\". "
            "или «ДЕЙСТВИТЕЛЬНЫЙ». Добавьте правило для преобразования этого типа заполнения в строку.\n"
            )
            self .run_tf =False
    def to_tensorflow (self ):
        assert self .output_size is not None
        tf_config =super (Conv2dTransposeConfig ,self ).to_tensorflow ()
        tf_config .filter_shape =[
        self .filter_size [0 ],self .filter_size [1 ],self .num_filters ,
        self .num_channels //self .groups
        ]
        if self .data_format =="NCHW":
            tf_config .output_size =[
            self .input_shape [0 ],self .num_filters ,self .output_size [0 ],
            self .output_size [1 ]
            ]
        elif self .data_format =="NHWC":
            tf_config .output_size =[
            self .input_shape [0 ],self .output_size [0 ],self .output_size [1 ],
            self .num_filters
            ]
        tf_config .padding =self ._convert_padding (self .padding )
        return tf_config
    def _convert_padding (self ,padding ):
        if isinstance (padding ,str ):
            return padding
        assert isinstance (padding ,list )
        return "VALID"if padding ==[0 ,0 ]else "SAME"
class PDConv2dTranspose (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        filter =self .variable (
        name ='filter',shape =config .filter_shape ,dtype =config .input_dtype )
        result =fluid .layers .conv2d_transpose (
        input =input ,
        num_filters =config .num_filters ,
        output_size =config .output_size ,
        filter_size =config .filter_size ,
        padding =config .padding ,
        stride =config .stride ,
        dilation =config .dilation ,
        groups =config .groups ,
        param_attr ="filter",
        bias_attr =False ,
        use_cudnn =config .use_cudnn ,
        act =None ,
        data_format =config .data_format )
        self .feed_vars =[input ,filter ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,filter ])
class TFConv2dTranspose (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        filter =self .variable (
        name ='filter',shape =config .filter_shape ,dtype =config .input_dtype )
        result =tf .nn .conv2d_transpose (
        input =input ,
        filters =filter ,
        output_shape =config .output_size ,
        strides =config .stride ,
        padding =config .padding ,
        data_format =config .data_format ,
        dilations =config .dilation )
        self .feed_list =[input ,filter ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,filter ])
if __name__ =='__main__':
    test_main (
    PDConv2dTranspose (),
    TFConv2dTranspose (),
    config =Conv2dTransposeConfig ())