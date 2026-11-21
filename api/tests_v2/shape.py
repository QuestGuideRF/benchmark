from common_import import *
class PDShape (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =paddle .shape (input =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFShape (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .shape (input =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDShape (),TFShape (),config =APIConfig ("shape"))