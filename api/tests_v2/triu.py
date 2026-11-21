from common_import import *
class TriuConfig (APIConfig ):
    def __init__ (self ):
        super (TriuConfig ,self ).__init__ ("triu")
        self .run_tf =False
class PDTriu (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .tensor .triu (x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFTriu (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .experimental .numpy .triu (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDTriu (),TFTriu (),config =TriuConfig ())