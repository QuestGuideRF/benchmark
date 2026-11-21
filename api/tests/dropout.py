from common_import import *
@benchmark_registry .register ("dropout")
class PaddleDropout (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .dropout (
        x =x ,p =config .p ,axis =config .axis ,mode =config .mode )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("dropout")
class TorchDropout (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        m =torch .nn .Dropout (p =config .p )
        result =m (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,self .feed_list )
@benchmark_registry .register ("dropout")
class TFDropout (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .dropout (x =x ,rate =config .p ,noise_shape =None )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])