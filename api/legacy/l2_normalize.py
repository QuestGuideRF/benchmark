from common_import import *
class PDL2Normalize (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .l2_normalize (
        x =x ,axis =config .axis ,epsilon =config .epsilon )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFL2Normalize (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .l2_normalize (
        x ,axis =config .axis ,epsilon =config .epsilon )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (
    PDL2Normalize (),TFL2Normalize (),config =APIConfig ("l2_normalize"))