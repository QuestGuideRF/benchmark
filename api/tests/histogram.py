from common_import import *
@benchmark_registry .register ("histogram")
class HistogramConfig (APIConfig ):
    def __init__ (self ):
        super (HistogramConfig ,self ).__init__ ("histogram")
        self .feed_spec =[{"range":[-20 ,20 ]}]
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (HistogramConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if not use_gpu ()and self .input_dtype in ["int32","int64"]:
            print (
            "Warning:\n"
            "1. Pytorch-CPU не может поддерживать гистограмму в настоящее время, как только dtype входных данных"
            " int32 или int64\n")
            self .run_torch =False
@benchmark_registry .register ("histogram")
class PaddleHistogram (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =paddle .histogram (
        input =x ,bins =config .bins ,min =config .min ,max =config .max )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("histogram")
class TorchHistogram (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =torch .histc (
        input =x ,bins =config .bins ,min =config .min ,max =config .max )
        self .feed_list =[x ]
        self .fetch_list =[result ]