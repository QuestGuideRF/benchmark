from common_import import *
class AnyConfig (APIConfig ):
    def __init__ (self ):
        super (AnyConfig ,self ).__init__ ('any')
        self .alias_name ="reduce_all_any"
class PDAny (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .any (x =x ,axis =config .axis ,keepdim =config .keepdim )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFAny (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .reduce_any (
        input_tensor =x ,axis =config .axis ,keepdims =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDAny (),TFAny (),config =AnyConfig ())