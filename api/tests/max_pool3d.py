from common_import import *
@benchmark_registry .register ("max_pool3d")
class PaddleMaxPool3d (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        if config .return_mask :
            result ,max_indices =paddle .nn .functional .max_pool3d (
            x =x ,
            kernel_size =config .kernel_size ,
            stride =config .stride ,
            padding =config .padding ,
            ceil_mode =config .ceil_mode ,
            return_mask =config .return_mask ,
            data_format =config .data_format )
            self .feed_list =[x ]
            self .fetch_list =[result ,max_indices ]
        else :
            result =paddle .nn .functional .max_pool3d (
            x =x ,
            kernel_size =config .kernel_size ,
            stride =config .stride ,
            padding =config .padding ,
            ceil_mode =config .ceil_mode ,
            return_mask =config .return_mask ,
            data_format =config .data_format )
            self .feed_list =[x ]
            self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("max_pool3d")
class TorchMaxPool3d (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        if config .return_mask :
            result ,max_indices =torch .nn .functional .max_pool3d (
            x ,
            kernel_size =config .kernel_size ,
            stride =config .stride ,
            padding =config .padding ,
            ceil_mode =config .ceil_mode ,
            return_indices =config .return_mask )
            self .feed_list =[x ]
            self .fetch_list =[result ,max_indices ]
        else :
            result =torch .nn .functional .max_pool3d (
            x ,
            kernel_size =config .kernel_size ,
            stride =config .stride ,
            padding =config .padding ,
            ceil_mode =config .ceil_mode ,
            return_indices =config .return_mask )
            self .feed_list =[x ]
            self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])