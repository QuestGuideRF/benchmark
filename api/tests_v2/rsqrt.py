from common_import import *
class RsqrtConfig (APIConfig ):
    def __init__ (self ):
        super (RsqrtConfig ,self ).__init__ ("rsqrt")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="activation"
class PDRsqrt (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .rsqrt (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFRsqrt (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .rsqrt (x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDRsqrt (),TFRsqrt (),config =RsqrtConfig ())