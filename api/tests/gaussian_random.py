from common_import import *
@benchmark_registry .register ("gaussian_random")
class GaussianRandomConfig (APIConfig ):
    def __init__ (self ):
        super (GaussianRandomConfig ,self ).__init__ ("gaussian_random")
        self .run_torch =False
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("gaussian_random")
class PaddleGaussianRandom (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =paddle .fluid .layers .gaussian_random (
        shape =config .shape ,
        mean =config .mean ,
        std =config .std ,
        seed =config .seed ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]