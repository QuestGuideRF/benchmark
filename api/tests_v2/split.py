from common_import import *
class PDSplit (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .split (
        x =data ,num_or_sections =config .num_or_sections ,axis =config .axis )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFSplit (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .split (
        value =data ,
        num_or_size_splits =config .num_or_sections ,
        axis =config .axis )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDSplit (),TFSplit (),config =APIConfig ("split"))