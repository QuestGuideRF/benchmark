from common_import import *
class AbsConfig (APIConfig ):
    def __init__ (self ):
        super (AbsConfig ,self ).__init__ ("abs")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="activation"
class PDAbs (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .abs (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFAbs (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .abs (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDAbs (),TFAbs (),config =AbsConfig ())