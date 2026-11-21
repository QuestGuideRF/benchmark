from common_import import *
class IsfiniteNanInfConfig (APIConfig ):
    def __init__ (self ):
        super (IsfiniteNanInfConfig ,self ).__init__ ("isfinite_nan_inf")
        self .api_name ='isfinite'
        self .api_list ={
        'isfinite':'is_finite',
        'isnan':'is_nan',
        'isinf':'is_inf'
        }
class PDIsfiniteNanInf (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
class TFIsfiniteNanInf (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[out ]
if __name__ =='__main__':
    test_main (
    PDIsfiniteNanInf (),TFIsfiniteNanInf (),config =IsfiniteNanInfConfig ())