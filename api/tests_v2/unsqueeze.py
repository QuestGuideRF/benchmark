from common_import import *
class PDUnsqueeze (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .unsqueeze (x =input ,axis =config .axis )
        self .feed_vars =[input ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
class TFUnsqueeze (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .expand_dims (input =input ,axis =config .axis )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
if __name__ =='__main__':
    test_main (PDUnsqueeze (),TFUnsqueeze (),config =APIConfig ("unsqueeze"))