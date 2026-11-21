from common_import import *
class CumsumConfig (APIConfig ):
    def __init__ (self ):
        super (CumsumConfig ,self ).__init__ ('cumsum')
        self .feed_spec ={"range":[-1 ,1 ]}
class PDCumsum (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .cumsum (
        x =x ,
        axis =config .axis ,
        exclusive =config .exclusive ,
        reverse =config .reverse )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFCumsum (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .cumsum (
        x =x ,
        axis =config .axis ,
        exclusive =config .exclusive ,
        reverse =config .reverse )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDCumsum (),TFCumsum (),config =CumsumConfig ())