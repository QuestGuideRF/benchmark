from common_import import *
class LayerNormConfig (APIConfig ):
    def __init__ (self ):
        super (LayerNormConfig ,self ).__init__ ('layer_norm')
        self .run_tf =False
class PDLayerNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .layer_norm (
        x =x ,normalized_shape =config .x_shape [1 :],epsilon =config .epsilon )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFLayerNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
if __name__ =='__main__':
    test_main (PDLayerNorm (),TFLayerNorm (),config =LayerNormConfig ())