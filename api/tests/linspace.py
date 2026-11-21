from common_import import *
@benchmark_registry .register ("linspace")
class PaddleLinspace (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =paddle .linspace (
        start =config .start ,
        stop =config .stop ,
        num =config .num ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("linspace")
class TorchLinspace (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =torch .linspace (
        start =config .start ,end =config .stop ,steps =config .num )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("linspace")
class TFLinspace (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .linspace (
        start =config .start ,stop =config .stop ,num =config .num )
        self .feed_list =[]
        self .fetch_list =[result ]