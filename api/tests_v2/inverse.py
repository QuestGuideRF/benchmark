from common_import import *
class PDInverse (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .inverse (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFInverse (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .linalg .inv (input =input )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
if __name__ =='__main__':
    test_main (PDInverse (),TFInverse (),config =APIConfig ('inverse'))