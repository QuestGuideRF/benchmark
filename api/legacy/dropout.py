from common_import import *
class PDDropout (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .dropout (
        x =x ,
        dropout_prob =config .dropout_prob ,
        seed =123 ,
        dropout_implementation =config .dropout_implementation )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFDropout (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .dropout (
        x =x ,rate =config .dropout_prob ,noise_shape =None ,seed =123 )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDDropout (),TFDropout (),config =APIConfig ("dropout"))