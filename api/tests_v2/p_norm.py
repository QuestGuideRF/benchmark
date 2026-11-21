from common_import import *
class PnormConfig (APIConfig ):
    def __init__ (self ):
        super (PnormConfig ,self ).__init__ ("p_norm")
        self .feed_spec =[{"range":[-1 ,1 ]}]
class PDPnorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .norm (
        x =x ,p =config .porder ,axis =config .axis ,keepdim =config .keepdim )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFPnorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .norm (
        tensor =x ,
        ord =config .porder ,
        axis =config .axis ,
        keepdims =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDPnorm (),TFPnorm (),config =PnormConfig ())