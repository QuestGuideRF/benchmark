from common_import import *
class SoftsignConfig (APIConfig ):
    def __init__ (self ):
        super (SoftsignConfig ,self ).__init__ ("softsign")
        self .alias_name ="activation"
class PDSoftsign (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .softsign (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
class TFSoftsign (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .softsign (features =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[data ])
if __name__ =='__main__':
    test_main (PDSoftsign (),TFSoftsign (),config =SoftsignConfig ())