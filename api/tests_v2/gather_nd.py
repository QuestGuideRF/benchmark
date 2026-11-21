from common_import import *
class GatherNdConfig (APIConfig ):
    def __init__ (self ):
        super (GatherNdConfig ,self ).__init__ ('gather_nd')
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
class PDGatherNd (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        index =self .variable (
        name ='index',
        shape =config .index_shape ,
        dtype =config .index_dtype ,
        stop_gradient =True )
        result =paddle .gather_nd (x =x ,index =index )
        self .feed_vars =[x ,index ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFGatherNd (TensorflowAPIBenchmarkBase ):
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
if __name__ =='__main__':
    test_main (PDGatherNd (),TFGatherNd (),config =GatherNdConfig ())