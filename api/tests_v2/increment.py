from common_import import *
class PDIncrement (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        x .stop_gradient =False
        result =paddle .increment (x ,value =config .value )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFAssignAdd (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        one_tensor =tf .Variable (
        tf .ones (
        shape =config .x_shape ,dtype =config .x_dtype ))
        result =tf .compat .v1 .assign_add (ref =x ,value =one_tensor )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDIncrement (),TFAssignAdd (),config =APIConfig ('increment'))