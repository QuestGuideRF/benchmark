from common_import import *
class DataNormConfig (APIConfig ):
    def __init__ (self ):
        super (DataNormConfig ,self ).__init__ ("data_norm")
        self .run_tf =False
class PDDataNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .static .nn .data_norm (input =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (pd_obj =PDDataNorm (),config =DataNormConfig ())