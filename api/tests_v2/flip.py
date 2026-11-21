from common_import import *
class PDFlip (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        dims =[]
        dims .append (config .axis )
        result =paddle .flip (x =x ,axis =dims )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFFlip (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        dims =[]
        dims .append (config .axis )
        result =tf .reverse (tensor =x ,axis =dims )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDFlip (),TFFlip (),config =APIConfig ("flip"))