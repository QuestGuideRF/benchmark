from common_import import *
@benchmark_registry .register ("index_sample")
class IndexSampleConfig (APIConfig ):
    def __init__ (self ):
        super (IndexSampleConfig ,self ).__init__ ("index_sample")
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (IndexSampleConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,self .x_shape [-1 ]]
        }
        ]
@benchmark_registry .register ("index_sample")
class PaddleIndexSample (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        index =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =paddle .index_sample (x =x ,index =index )
        self .feed_list =[x ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("index_sample")
class TorchIndexSample (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        index =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =torch .gather (input =x ,index =index ,dim =1 )
        self .feed_list =[x ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])