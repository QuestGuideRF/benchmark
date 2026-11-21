from common_import import *
class PDIsfinite (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .isfinite (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFIsfinite (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .is_finite (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDIsfinite (),TFIsfinite (),config =APIConfig ("isfinite"))