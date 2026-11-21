from common_import import *
@benchmark_registry .register ("adaptive_avg_pool2d")
class PaddleAdaptiveAvgPool2D (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        adaptive_avg_pool2d =paddle .nn .AdaptiveAvgPool2D (
        output_size =config .output_size ,data_format =config .data_format )
        result =adaptive_avg_pool2d (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("adaptive_avg_pool2d")
class TorchAdaptiveAvgPool2D (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        adaptive_avg_pool2d =torch .nn .AdaptiveAvgPool2d (
        output_size =config .output_size )
        result =adaptive_avg_pool2d (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("adaptive_avg_pool2d")
class TFAdaptiveAvgPool2d (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        import tensorflow_addons as tfa
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        data_format ="channels_first"
        if config .data_format =='NHWC':
            data_format ="channels_last"
        tf_func =tfa .layers .AdaptiveAveragePooling2D (
        output_size =config .output_size ,data_format =data_format )
        out =tf_func (x )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])