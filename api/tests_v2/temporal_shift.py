from common_import import *
class TemporalShiftConfig (APIConfig ):
    def __init__ (self ):
        super (TemporalShiftConfig ,self ).__init__ ("temporal_shift")
        self .run_tf =False
class PDTemporalShift (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .temporal_shift (
        x =x ,seg_num =config .seg_num ,shift_ratio =config .shift_ratio )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDTemporalShift (),config =TemporalShiftConfig ())