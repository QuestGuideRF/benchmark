from common_import import *
class AffineGridConfig (APIConfig ):
    def __init__ (self ):
        super (AffineGridConfig ,self ).__init__ ("affine_grid")
        self .run_tf =False
class PDAffineGrid (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        theta =self .variable (
        name ='theta',shape =config .theta_shape ,dtype =config .theta_dtype )
        out =paddle .nn .functional .affine_grid (
        theta ,
        out_shape =config .out_shape ,
        align_corners =config .align_corners )
        self .feed_vars =[theta ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[theta ])
if __name__ =='__main__':
    test_main (PDAffineGrid (),config =AffineGridConfig ())