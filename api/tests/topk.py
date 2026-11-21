from common_import import *
@benchmark_registry .register ("topk")
class PaddleTopK (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =paddle .topk (x =x ,k =config .k )
        self .feed_list =[x ]
        self .fetch_list =[value ,indices ]
@benchmark_registry .register ("topk")
class TorchTopK (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =torch .topk (input =x ,k =config .k )
        self .feed_list =[x ]
        self .fetch_list =[value ,indices ]
@benchmark_registry .register ("topk")
class TFTopK (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =tf .math .top_k (input =x ,k =config .k )
        self .feed_list =[x ]
        self .fetch_list =[value ,indices ]
        if config .backward :
            self .append_gradients ([value ],[x ])