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
    def disabled (self ):
        if self .input_dtype =="float16"and self .use_cudnn ==False :
            print (
            "Warning:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s, когда use_cudnn имеет значение False.\n"
            %(self .api_name ))
            return True
        return super (Conv2dConfig ,self ).disabled ()
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
            self .num_channels =self .input_shape [1 ]
        elif self .data_format =="NHWC":
            self .num_channels =self .input_shape [3 ]
        if isinstance (self .filter_size ,int ):
            self .filter_size =[self .filter_size ,self .filter_size ]
        if self .groups is None :
            self .groups =1
        if self .num_channels %self .groups !=0 :
            raise ValueError (
            "канал ввода должен делиться на группы,"
            "получено: канал ввода — {}, форма ввода — {}"
            ", группы {}".format (self .num_channels ,
            self .input_shape ,self .groups ))
        self .filter_shape =[
        self .num_filters ,self .num_channels //self .groups ,
        self .filter_size [0 ],self .filter_size [1 ]
        ]
    def to_tensorflow (self ):
        tf_config =super (Conv2dConfig ,self ).to_tensorflow ()
        tf_config .filter_shape =[
        self .filter_size [0 ],self .filter_size [1 ],
        self .num_channels //self .groups ,self .num_filters
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
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        filter =self .variable (
        name ='filter',shape =config .filter_shape ,dtype =config .input_dtype )
        result =fluid .layers .conv2d (
        input =input ,
        num_filters =config .num_filters ,
        filter_size =config .filter_size ,
        stride =config .stride ,
        padding =config .padding ,
        dilation =config .dilation ,
        groups =config .groups ,
        param_attr ='filter',
        bias_attr =False ,
        use_cudnn =config .use_cudnn ,
        act =None ,
        data_format =config .data_format )
        self .feed_vars =[input ,filter ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,filter ])
class TFConv2d (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        filter =self .variable (
        name ='filter',shape =config .filter_shape ,dtype =config .input_dtype )
        if tf .__version__ <="1.15.0":
            result =tf .nn .conv2d (
            input =input ,
            filters =filter ,
            strides =config .stride ,
            padding =config .padding ,
            data_format =config .data_format ,
            dilations =config .dilation ,
            use_cudnn_on_gpu =config .use_cudnn )
        else :
            result =tf .nn .conv2d (
            input =input ,
            filters =filter ,
            strides =config .stride ,
            padding =config .padding ,
            data_format =config .data_format ,
            dilations =config .dilation )
        self .feed_list =[input ,filter ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,filter ])
if __name__ =='__main__':
    test_main (PDConv2d (),TFConv2d (),config =Conv2dConfig ())