from common_import import *
class InterpAreaConfig (APIConfig ):
    def __init__ (self ):
        super (InterpAreaConfig ,self ).__init__ ("interp_area")
        self .run_tf =False
class PDInterpArea (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="area",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDInterpArea (),config =InterpAreaConfig ())