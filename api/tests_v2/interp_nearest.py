from common_import import *
class InterpNearestConfig (APIConfig ):
    def __init__ (self ):
        super (InterpNearestConfig ,self ).__init__ ("interp_nearest")
        self .run_tf =False
class PDInterpNearest (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x =x ,
        size =config .size ,
        mode ="nearest",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDInterpNearest (),config =InterpNearestConfig ())