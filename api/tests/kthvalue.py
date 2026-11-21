from common_import import *
@benchmark_registry .register ("kthvalue")
class PaddleKthvalue (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =paddle .kthvalue (
        x =data ,k =config .k ,keepdim =config .keepdim )
        self .feed_list =[data ]
        self .fetch_list =[value ,indices ]
        if config .backward :
            self .append_gradients (value ,[data ])
@benchmark_registry .register ("kthvalue")
class TorchKthvalue (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =torch .kthvalue (
        input =data ,k =config .k ,keepdim =config .keepdim )
        self .feed_list =[data ]
        self .fetch_list =[value ,indices ]
        if config .backward :
            self .append_gradients (value ,[data ])