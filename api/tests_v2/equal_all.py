from common_import import *
class EqualAllConfig (APIConfig ):
    def __init__ (self ):
        super (EqualAllConfig ,self ).__init__ ("equal_all")
        self .run_tf =False
class PDEqualAll (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data_x =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        data_y =self .variable (
        name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        out =paddle .equal_all (x =data_x ,y =data_y )
        self .feed_vars =[data_x ,data_y ]
        self .fetch_vars =[out ]
if __name__ =='__main__':
    test_main (PDEqualAll (),config =EqualAllConfig ())