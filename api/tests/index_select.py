from common_import import *
@benchmark_registry .register ("index_select")
class IndexSelectConfig (APIConfig ):
    def __init__ (self ):
        super (IndexSelectConfig ,self ).__init__ ('index_select')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (IndexSelectConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        dim =self .axis
        self .feed_spec =[
        {
        "range":[0 ,10 ]
        },
        {
        "range":[0 ,self .x_shape [dim ]]
        }
        ]
@benchmark_registry .register ("index_select")
class PaddleIndexSelect (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        idx =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =paddle .index_select (x =x ,index =idx ,axis =config .axis )
        self .feed_list =[x ,idx ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("index_select")
class TorchIndexSelect (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        idx =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =torch .index_select (input =x ,dim =config .axis ,index =idx )
        self .feed_list =[x ,idx ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])