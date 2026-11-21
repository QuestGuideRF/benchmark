from common_import import *
class DiagConfig (APIConfig ):
    def __init__ (self ):
        super (DiagConfig ,self ).__init__ ("diag")
class PDDiag (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .diag (
        x =x ,offset =config .offset ,padding_value =config .padding_value )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFDiag (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .linalg .diag (
        diagonal =x ,k =config .offset ,padding_value =config .padding_value )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDDiag (),TFDiag (),config =DiagConfig ())