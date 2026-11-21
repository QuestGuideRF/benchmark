from common_import import *
@benchmark_registry .register ("graph_send_recv")
class GraphSendRecvConfig (APIConfig ):
    def __init__ (self ):
        super (GraphSendRecvConfig ,self ).__init__ ('graph_send_recv')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (GraphSendRecvConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[1 ,self .x_shape [0 ]]
        },
        {
        "range":[1 ,self .x_shape [0 ]]
        }
        ]
@benchmark_registry .register ("graph_send_recv")
class PaddleGraphSendRecv (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        src_index =self .variable (
        name ='src_index',
        shape =config .src_index_shape ,
        dtype =config .src_index_dtype )
        dst_index =self .variable (
        name ='dst_index',
        shape =config .dst_index_shape ,
        dtype =config .dst_index_dtype )
        result =paddle .incubate .graph_send_recv (
        x ,src_index ,dst_index ,pool_type =config .pool_type )
        self .feed_vars =[x ,src_index ,dst_index ]
        self .fetch_vars =[result ]