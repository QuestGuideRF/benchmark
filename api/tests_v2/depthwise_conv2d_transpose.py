from common_import import *
class DepthwiseConv2dTransposeConfig (APIConfig ):
    def __init__ (self ):
        super (DepthwiseConv2dTransposeConfig ,
        self ).__init__ ("depthwise_conv2d_transpose")
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        }
        ]
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (DepthwiseConv2dTransposeConfig ,self ).init_from_json (
        filename ,config_id ,unknown_dim )
        if isinstance (self .padding ,int ):
            self .padding =[self .padding ,self .padding ]
        if self .data_format =="NCHW":
            self .num_channels =self .input_shape [1 ]
        elif self .data_format =="NHWC":
            self .num_channels =self .input_shape [3 ]
        if isinstance (self .filter_size ,int ):
            self .filter_size =[self .filter_size ,self .filter_size ]
        assert self .num_filters ==self .groups and self .num_channels !=1
        self .filter_shape =[
        self .num_channels ,self .num_filters //self .groups ,
        self .filter_size [0 ],self .filter_size [1 ]
        ]
        self .run_tf =False
class PDDepthwiseConv2dTranspose (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        filter =self .variable (
        name ='filter',shape =config .filter_shape ,dtype =config .input_dtype )
        result =paddle .nn .functional .conv2d_transpose (
        x =input ,
        weight =filter ,
        output_size =config .output_size ,
        stride =config .stride ,
        padding =config .padding ,
        data_format =config .data_format ,
        groups =config .groups ,
        dilation =config .dilation )
        self .feed_vars =[input ,filter ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,filter ])
if __name__ =='__main__':
    test_main (
    PDDepthwiseConv2dTranspose (),config =DepthwiseConv2dTransposeConfig ())