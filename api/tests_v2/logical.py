from common_import import *
class LogicalConfig (APIConfig ):
    def __init__ (self ):
        super (LogicalConfig ,self ).__init__ ('logical')
        self .api_name ='logical_and'
        self .api_list ={
        'logical_and':'logical_and',
        'logical_or':'logical_or'
        }
class PDLogical (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
class TFLogical (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =self .layers (config .api_name ,x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDLogical (),TFLogical (),config =LogicalConfig ())