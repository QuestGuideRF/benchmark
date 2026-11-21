from common_import import *
class MeanConfig (APIConfig ):
    def __init__ (self ):
        super (MeanConfig ,self ).__init__ ('mean')
class PDMean (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .mean (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFMean (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        _ ,result =tf .compat .v1 .metrics .mean (values =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDMean (),TFMean (),config =MeanConfig ())