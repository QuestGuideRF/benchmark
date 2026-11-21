from common_import import *
class PDSplit (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .layers .split (
        input =data ,num_or_sections =config .num_or_sections ,dim =config .dim )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFSplit (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .split (
        value =data ,
        num_or_size_splits =config .num_or_sections ,
        axis =config .dim )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDSplit (),TFSplit (),config =APIConfig ("split"))