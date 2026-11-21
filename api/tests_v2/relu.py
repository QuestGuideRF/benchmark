from common_import import *
class ReluConfig (APIConfig ):
    def __init__ (self ):
        super (ReluConfig ,self ).__init__ ("relu")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .api_list ={'relu':'relu','relu6':'relu6','elu':'elu'}
        self .alias_name ="activation"
class PDRelu (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
class TFRelu (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,features =x )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDRelu (),TFRelu (),config =ReluConfig ())