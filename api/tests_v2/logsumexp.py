from common_import import *
class LogsumexpConfig (APIConfig ):
    def __init__ (self ):
        super (LogsumexpConfig ,self ).__init__ ('logsumexp')
class PDLogsumexp (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .logsumexp (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFLogsumexp (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .reduce_logsumexp (input_tensor =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDLogsumexp (),TFLogsumexp (),config =LogsumexpConfig ())