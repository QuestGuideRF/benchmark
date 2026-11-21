from common_import import *
@benchmark_registry .register ("gather")
class GatherConfig (APIConfig ):
    def __init__ (self ):
        super (GatherConfig ,self ).__init__ ('gather')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (GatherConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[8 ,10 ]
        },
        {
        "range":[1 ,self .x_shape [self .axis ]]
        }
        ]
        if len (self .index_shape )>1 :
            self .run_torch =False
@benchmark_registry .register ("gather")
class PaddleGather (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        index =self .variable (
        name ='index',
        shape =config .index_shape ,
        dtype =config .index_dtype ,
        stop_gradient =True )
        result =paddle .gather (x =x ,index =index ,axis =config .axis )
        self .feed_list =[x ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("gather")
class TorchGather (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        index =self .variable (
        name ='index',shape =config .index_shape ,dtype ="int64")
        result =torch .index_select (x =x ,index =index ,dim =config .axis )
        self .feed_list =[x ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("gather")
class TFGather (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        params =self .variable (
        name ='params',shape =config .x_shape ,dtype =config .x_dtype )
        indices =self .variable (
        name ='indices',shape =config .index_shape ,dtype =config .index_dtype )
        result =tf .gather (params =params ,indices =indices ,axis =config .axis )
        self .feed_list =[params ,indices ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[params ])