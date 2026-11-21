from common_import import *
@benchmark_registry .register ("log_softmax")
class LogSoftmaxConfig (APIConfig ):
    def __init__ (self ):
        super (LogSoftmaxConfig ,self ).__init__ ("log_softmax")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="softmax"
        self .atol =1E-5
@benchmark_registry .register ("log_softmax")
class PaddleLogSoftmax (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .log_softmax (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        softmax_dim =x_shape [config .axis ]
        forward_flop =numel (x_shape )*5 +numel (x_shape )/softmax_dim
        forward_byte =numel (x_shape )*2 *sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("log_softmax")
class TorchLogSoftmax (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =torch .nn .functional .log_softmax (input =x ,dim =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
@benchmark_registry .register ("log_softmax")
class TFLogSoftmax (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =tf .nn .log_softmax (logits =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])