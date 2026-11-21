from common_import import *
class ReluConfig (APIConfig ):
    def __init__ (self ):
        super (ReluConfig ,self ).__init__ ("relu")
        self .alias_name ="activation"
class PDRelu (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .relu (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFRelu (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .relu (features =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDRelu (),TFRelu (),config =ReluConfig ())