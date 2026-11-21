from common_import import *
class UniqueConfig (APIConfig ):
    def __init__ (self ):
        super (UniqueConfig ,self ).__init__ ("unique")
        self .run_tf =False
class PDUnique (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .unique (
        x =x ,
        return_index =config .return_index ,
        return_inverse =config .return_inverse ,
        return_counts =config .return_counts ,
        axis =config .axis ,
        dtype =config .dtype )
        if isinstance (result ,tuple ):
            result =list (result )
        else :
            result =[result ]
        self .feed_vars =[x ]
        self .fetch_vars =result
if __name__ =='__main__':
    test_main (pd_obj =PDUnique (),config =UniqueConfig ())