from common_import import *
class LogSoftmaxConfig (APIConfig ):
    def __init__ (self ):
        super (LogSoftmaxConfig ,self ).__init__ ("log_softmax")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="softmax"
        self .atol =1E-5
class PDLogSoftmax (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .log_softmax (x =x ,axis =config .axis )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
class TFLogSoftmax (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =tf .nn .log_softmax (logits =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDLogSoftmax (),TFLogSoftmax (),config =LogSoftmaxConfig ())