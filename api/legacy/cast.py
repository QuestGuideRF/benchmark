from common_import import *
class CastConfig (APIConfig ):
    def __init__ (self ):
        super (CastConfig ,self ).__init__ ('cast')
        self .feed_spec ={"range":[-10 ,10 ]}
class PDCast (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .cast (x =data ,dtype =config .dtype )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFCast (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .dtypes .cast (x =data ,dtype =config .dtype )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDCast (),TFCast (),config =CastConfig ())