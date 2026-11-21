from common_import import *
@benchmark_registry .register ("gather_nd")
class GatherNdConfig (APIConfig ):
    def __init__ (self ):
        super (GatherNdConfig ,self ).__init__ ('gather_nd')
        self .run_torch =False
        print ("[ВНИМАНИЕ]: Pytorch в настоящее время не поддерживает сбор_nd.")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (GatherNdConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,10 ]
        },
        {
        "range":[1 ,int (min (self .input_shape ))]
        }
        ]
@benchmark_registry .register ("gather_nd")
class PaddleGatherNd (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        index =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =paddle .gather_nd (x =x ,index =index )
        self .feed_list =[x ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("gather_nd")
class TFGatherNd (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        params =self .variable (
        name ='params',shape =config .input_shape ,dtype =config .input_dtype )
        indices =self .variable (
        name ='indices',shape =config .index_shape ,dtype =config .index_dtype )
        result =tf .gather_nd (params =params ,indices =indices )
        self .feed_list =[params ,indices ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[params ])