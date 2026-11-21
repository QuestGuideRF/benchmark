from common_import import *
class ArgConfig (APIConfig ):
    def __init__ (self ):
        super (ArgConfig ,self ).__init__ ('arg')
        self .api_name ='argmax'
        self .api_list ={'argmax':'argmax','argmin':'argmin'}
class PDArg (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .fluid_layers (config .api_name ,x =x ,axis =config .axis )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFArg (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,input =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDArg (),TFArg (),config =ArgConfig ())