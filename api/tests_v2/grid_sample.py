from common_import import *
class GridSampleConfig (APIConfig ):
    def __init__ (self ):
        super (GridSampleConfig ,self ).__init__ ("grid_sample")
        self .run_tf =False
class PDGridSample (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        grid =self .variable (
        name ='grid',shape =config .grid_shape ,dtype =config .grid_dtype )
        out =paddle .nn .functional .grid_sample (
        x ,
        grid ,
        mode =config .mode ,
        padding_mode =config .padding_mode ,
        align_corners =config .align_corners )
        self .feed_vars =[x ,grid ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ,grid ])
if __name__ =='__main__':
    test_main (PDGridSample (),config =GridSampleConfig ())