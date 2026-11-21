from common_import import *
class GroupNormConfig (APIConfig ):
    def __init__ (self ):
        super (GroupNormConfig ,self ).__init__ ('group_norm')
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (GroupNormConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .x_shape )==4 :
            self .num_channels =self .x_shape [1 ]if self .data_format =="NCHW"else self .x_shape [3 ]
        else :
            self .num_channels =self .x_shape [1 ]
    def to_tensorflow (self ):
        tf_config =super (GroupNormConfig ,self ).to_tensorflow ()
        if len (self .x_shape )==4 :
            tf_config .channel_axis =-3 if self .data_format =="NCHW"else -1
        return tf_config
class PDGroupNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        paddle_group_norm =paddle .nn .GroupNorm (
        num_channels =config .num_channels ,
        num_groups =config .num_groups ,
        epsilon =config .epsilon )
        result =paddle_group_norm (x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFGroupNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
if __name__ =='__main__':
    test_main (PDGroupNorm (),TFGroupNorm (),config =GroupNormConfig ())