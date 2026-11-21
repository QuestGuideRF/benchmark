from common_import import *
@benchmark_registry .register ("softmax")
class SoftmaxConfig (APIConfig ):
    def __init__ (self ):
        super (SoftmaxConfig ,self ).__init__ ("softmax")
        self .feed_spec ={"range":[-1 ,1 ]}
@benchmark_registry .register ("softmax")
class PaddleSoftmax (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .softmax (x =x ,axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        forward_flop =numel (x_shape )*5
        forward_byte =numel (x_shape )*2 *sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("softmax")
class TorchSoftmax (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .softmax (input =x ,dim =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("softmax")
class TFSoftmax (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .softmax (logits =input ,axis =config .axis )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])