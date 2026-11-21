from common_import import *
class ArgsortConfig (APIConfig ):
    def __init__ (self ):
        super (ArgsortConfig ,self ).__init__ ("argsort")
    def to_tensorflow (self ):
        tf_config =self
        if self .descending :
            tf_config .direction ="DESCENDING"
        else :
            tf_config .direction ="ASCENDING"
        return tf_config
class PDArgsort (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result ,indices =fluid .layers .argsort (
        input =input ,axis =config .axis ,descending =config .descending )
        self .feed_vars =[input ]
        self .fetch_vars =[indices ]
class TFArgsort (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        indices =tf .argsort (
        values =input ,
        axis =config .axis ,
        direction =config .direction ,
        stable =False )
        self .feed_list =[input ]
        self .fetch_list =[indices ]
if __name__ =='__main__':
    test_main (PDArgsort (),TFArgsort (),config =ArgsortConfig ())