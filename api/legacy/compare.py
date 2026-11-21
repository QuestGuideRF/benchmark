from common_import import *
class CompareConfig (APIConfig ):
    def __init__ (self ):
        super (CompareConfig ,self ).__init__ ('compare')
        self .api_name ='less_than'
        self .api_list ={
        'less_than':'less',
        'less_equal':'less_equal',
        'not_equal':'not_equal',
        'greater_than':'greater',
        'greater_equal':'greater_equal',
        'equal':'equal'
        }
class PDCompare (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .fluid_layers (config .api_name ,x =x ,y =y )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
class TFCompare (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDCompare (),TFCompare (),config =CompareConfig ())