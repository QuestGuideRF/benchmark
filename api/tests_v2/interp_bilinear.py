from common_import import *
class InterpBilinearConfig (APIConfig ):
    def __init__ (self ):
        super (InterpBilinearConfig ,self ).__init__ ("interp_bilinear")
        self .run_tf =False
class PDInterpBilinear (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="bilinear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDInterpBilinear (),config =InterpBilinearConfig ())