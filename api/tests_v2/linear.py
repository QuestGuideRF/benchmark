from common_import import *
class LinearConfig (APIConfig ):
    def __init__ (self ):
        super (LinearConfig ,self ).__init__ ('linear')
    def to_tensorflow (self ):
        tf_config =super (LinearConfig ,self ).to_tensorflow ()
        tf_config .size =self .weight_shape [-1 ]
        return tf_config
class PDLinear (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        weight =paddle .create_parameter (
        shape =config .weight_shape ,
        dtype =config .weight_dtype ,
        name ="weight",
        attr =paddle .ParamAttr (
        initializer =paddle .nn .initializer .Constant (0.5 )))
        bias =paddle .create_parameter (
        shape =config .bias_shape ,
        dtype =config .bias_dtype ,
        name ="bias",
        attr =paddle .ParamAttr (
        initializer =paddle .nn .initializer .Constant (0.1 )))
        result =paddle .nn .functional .linear (x =x ,weight =weight ,bias =bias )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFLinear (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .compat .v1 .layers .dense (
        inputs =x ,
        units =config .size ,
        activation =None ,
        use_bias =True ,
        kernel_initializer =tf .constant_initializer (0.5 ),
        bias_initializer =tf .constant_initializer (0.1 ))
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDLinear (),TFLinear (),config =LinearConfig ())