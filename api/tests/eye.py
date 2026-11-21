from common_import import *
@benchmark_registry .register ("eye")
class PaddleEye (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        result =paddle .eye (num_rows =config .num_rows ,
        num_columns =config .num_columns ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
@benchmark_registry .register ("eye")
class TorchEye (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        dtype_map ={
        "float16":torch .float16 ,
        "float32":torch .float32 ,
        "float64":torch .float64 ,
        "int32":torch .int32 ,
        "int64":torch .int64 ,
        "bool":torch .bool
        }
        result =torch .eye (n =config .num_rows ,
        m =config .num_columns ,
        dtype =dtype_map [config .dtype ])
        self .feed_list =[]
        self .fetch_list =[result ]