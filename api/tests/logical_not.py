from common_import import *
@benchmark_registry .register ("logical_not")
class PaddleLogicalNot (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .logical_not (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("logical_not")
class TorchLogicalNot (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .logical_not (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("logical_not")
class TFLogicalNot (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .logical_not (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]