from common_import import *
class InstanceNormConfig (APIConfig ):
    def __init__ (self ):
        super (InstanceNormConfig ,self ).__init__ ('instance_norm')
        self .run_tf =False
class PDInstanceNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .layers .instance_norm (input =data ,epsilon =config .epsilon )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFInstanceNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
if __name__ =='__main__':
    test_main (PDInstanceNorm (),TFInstanceNorm (),config =InstanceNormConfig ())