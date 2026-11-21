from common_import import *
class InstanceNormConfig (APIConfig ):
    def __init__ (self ):
        super (InstanceNormConfig ,self ).__init__ ('instance_norm')
        self .run_tf =False
class PDInstanceNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .instance_norm (x =x ,eps =config .eps )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFInstanceNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
if __name__ =='__main__':
    test_main (PDInstanceNorm (),TFInstanceNorm (),config =InstanceNormConfig ())