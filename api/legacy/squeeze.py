from common_import import *
class PDSqueeze (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .layers .squeeze (input =data ,axes =config .axes )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFSqueeze (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .squeeze (input =data ,axis =config .axes )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDSqueeze (),TFSqueeze (),config =APIConfig ("squeeze"))