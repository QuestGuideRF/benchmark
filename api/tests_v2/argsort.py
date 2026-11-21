from common_import import *
class ArgsortConfig (APIConfig ):
    def __init__ (self ):
        super (ArgsortConfig ,self ).__init__ ("argsort")
    def to_tensorflow (self ):
        tf_config =super (ArgsortConfig ,self ).to_tensorflow ()
        if self .descending :
            tf_config .direction ="DESCENDING"
        else :
            tf_config .direction ="ASCENDING"
        return tf_config
class PDArgsort (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        indices =paddle .argsort (
        x =x ,axis =config .axis ,descending =config .descending )
        self .feed_vars =[x ]
        self .fetch_vars =[indices ]
class TFArgsort (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        indices =tf .argsort (
        values =x ,
        axis =config .axis ,
        direction =config .direction ,
        stable =False )
        self .feed_list =[x ]
        self .fetch_list =[indices ]
if __name__ =='__main__':
    test_main (PDArgsort (),TFArgsort (),config =ArgsortConfig ())