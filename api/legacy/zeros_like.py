from common_import import *
class PDZerosLike (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .zeros_like (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFZerosLike (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .zeros_like (input =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDZerosLike (),TFZerosLike (),config =APIConfig ("zeros_like"))