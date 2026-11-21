from common_import import *
@benchmark_registry .register ("avg_pool3d")
class PaddleAvgPool3d (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .avg_pool3d (
        x =x ,
        kernel_size =config .kernel_size ,
        stride =config .stride ,
        padding =config .padding ,
        ceil_mode =config .ceil_mode ,
        data_format =config .data_format )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("avg_pool3d")
class TorchAvgPool3d (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .avg_pool3d (
        input =x ,
        kernel_size =config .kernel_size ,
        stride =config .stride ,
        padding =config .padding ,
        ceil_mode =config .ceil_mode )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])