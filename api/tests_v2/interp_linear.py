from common_import import *
class InterpLinearConfig (APIConfig ):
    def __init__ (self ):
        super (InterpLinearConfig ,self ).__init__ ("interp_linear")
        self .run_tf =False
class PDInterpLinear (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .interpolate (
        x ,
        size =config .size ,
        mode ="linear",
        align_corners =config .align_corners ,
        scale_factor =config .scale_factor ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDInterpLinear (),config =InterpLinearConfig ())