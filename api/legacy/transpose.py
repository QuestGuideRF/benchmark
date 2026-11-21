from common_import import *
class PDTranspose (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .transpose (x =data ,perm =config .perm )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFTranspose (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .transpose (a =data ,perm =config .perm )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDTranspose (),TFTranspose (),config =APIConfig ("transpose"))