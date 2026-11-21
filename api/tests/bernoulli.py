from common_import import *
@benchmark_registry .register ("bernoulli")
class PaddleBernoulli (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .bernoulli (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("bernoulli")
class TorchBernoulli (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .bernoulli (input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("bernoulli")
class TFBernoulli (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        import tensorflow_probability as tfp
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        b =tfp .distributions .Bernoulli (probs =x )
        result =b .sample ()
        self .feed_list =[x ]
        self .fetch_list =[result ]