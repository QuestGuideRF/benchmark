from common_import import *
class PDReshape (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .reshape (x =data ,shape =config .shape )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFReshape (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .reshape (tensor =data ,shape =config .shape )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDReshape (),TFReshape (),config =APIConfig ("reshape"))