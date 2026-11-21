from common_import import *
class ClipByNormConfig (APIConfig ):
    def __init__ (self ):
        super (ClipByNormConfig ,self ).__init__ ("clip_by_norm")
        self .feed_spec ={"range":[-10 ,10 ]}
class PDClipByNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .clip_by_norm (x =x ,max_norm =2.0 )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFClipByNorm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .clip_by_norm (t =x ,clip_norm =2.0 ,axes =None )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDClipByNorm (),TFClipByNorm (),config =ClipByNormConfig ())