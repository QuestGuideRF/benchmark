from common_import import *
@benchmark_registry .register ("arange")
class PaddleArange (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =paddle .arange (
        start =config .start ,
        end =config .end ,
        step =config .step ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("arange")
class TorchArange (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =torch .range (
        start =config .start ,
        end =config .end -config .step ,
        step =config .step ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("arange")
class TFArange (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .range (
        start =config .start ,
        limit =config .end ,
        delta =config .step ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]