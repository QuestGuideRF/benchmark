from common_import import *
@benchmark_registry .register ("full")
class PaddleFull (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =paddle .full (
        shape =config .shape ,
        fill_value =config .fill_value ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("full")
class TorchFull (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        torch_list =[]
        torch_tensor =torch .tensor (config .shape ,dtype =torch .float32 )
        result =torch .nn .init .constant_ (
        tensor =torch_tensor ,val =config .fill_value )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("full")
class TFFull (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .constant (
        shape =config .shape ,
        dtype =tf .as_dtype (config .dtype ),
        value =config .fill_value )
        self .feed_list =[]
        self .fetch_list =[result ]