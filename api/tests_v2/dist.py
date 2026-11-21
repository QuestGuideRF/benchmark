from common_import import *
class DistConfig (APIConfig ):
    def __init__ (self ):
        super (DistConfig ,self ).__init__ ("dist")
        self .run_tf =False
class PDDist (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .dist (x =x ,y =y ,p =config .p )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
if __name__ =='__main__':
    test_main (PDDist (),config =DistConfig ())