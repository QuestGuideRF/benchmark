from common_import import *
@benchmark_registry .register ("log_loss")
class LogLossConfig (APIConfig ):
    def __init__ (self ):
        super (LogLossConfig ,self ).__init__ ("log_loss")
        self .run_torch =False
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
@benchmark_registry .register ("log_loss")
class PaddleLogLoss (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        label =self .variable (
        name ='label',shape =config .label_shape ,dtype =config .label_dtype )
        result =paddle .nn .functional .log_loss (
        input =input ,label =label ,epsilon =config .epsilon )
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])