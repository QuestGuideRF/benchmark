from common_import import *
class LayerNormConfig (APIConfig ):
    def __init__ (self ):
        super (LayerNormConfig ,self ).__init__ ('layer_norm')
        self .run_tf =False
class PDLayerNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .layers .layer_norm (
        input =data ,
        scale =config .scale ,
        shift =config .shift ,
        begin_norm_axis =config .begin_norm_axis ,
        epsilon =config .epsilon ,
        act =None )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFLayerNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
if __name__ =='__main__':
    test_main (PDLayerNorm (),TFLayerNorm (),config =LayerNormConfig ())