from common_import import *
class HistogramConfig (APIConfig ):
    def __init__ (self ):
        super (HistogramConfig ,self ).__init__ ("histogram")
        self .run_tf =False
class PDHistogram (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        out =paddle .histogram (input =input ,bins =config .bins ,min =config .min ,max =config .max )
        self .feed_vars =[input ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[input ])
if __name__ =='__main__':
    test_main (pd_obj =PDHistogram (),config =HistogramConfig ())