from common_import import *
@benchmark_registry .register ("truncated_gaussian_random")
class TruncatedGaussianRandomConfig (APIConfig ):
    def __init__ (self ):
        super (TruncatedGaussianRandomConfig ,
        self ).__init__ ("truncated_gaussian_random")
        self .run_torch =False
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("truncated_gaussian_random")
class PaddleTruncatedGaussianRandom (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        paddle .seed (config .seed )
        normal =paddle .nn .initializer .TruncatedNormal (
        mean =config .mean ,std =config .std )
        w =self .variable (name ="w",shape =config .w_shape ,dtype =config .w_dtype )
        result =normal (w )
        self .feed_list =[]
        self .fetch_list =[result ]