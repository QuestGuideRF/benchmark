from common_import import *
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
class PDIndexSelect (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        index =self .variable (
        name ='index',shape =config .index_shape ,dtype =config .index_dtype )
        result =paddle .index_select (x =x ,index =index ,axis =config .axis )
        self .feed_vars =[x ,index ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDIndexSelect (),config =IndexSelectConfig ())