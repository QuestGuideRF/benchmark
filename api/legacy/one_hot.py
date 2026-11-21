from common_import import *
class OneHotConfig (APIConfig ):
    def __init__ (self ):
        super (OneHotConfig ,self ).__init__ ('one_hot')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (OneHotConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec ={"range":[0 ,self .depth ]}
class PDOneHot (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .one_hot (input =data ,depth =config .depth )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFOneHot (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .one_hot (
        indices =data ,
        depth =config .depth ,
        on_value =None ,
        off_value =None ,
        axis =None ,
        dtype =None )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDOneHot (),TFOneHot (),config =OneHotConfig ())