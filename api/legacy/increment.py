from common_import import *
class PDIncrement (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        with fluid .program_guard (self .main_program ,self .startup_program ):
            x =self .variable (
            name ='x',shape =config .x_shape ,dtype =config .x_dtype )
            x .stop_gradient =False
            result =fluid .layers .increment (x ,1 )
            self .feed_vars =[x ]
            self .fetch_vars =[result ]
            if config .backward :
                self .append_gradients (result ,[x ])
class TFAssignAdd (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        one_tensor =tf .Variable (
        tf .ones (
        shape =config .x_shape ,dtype =config .x_dtype ))
        result =tf .compat .v1 .assign_add (ref =x ,value =one_tensor )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDIncrement (),TFAssignAdd (),config =APIConfig ('increment'))