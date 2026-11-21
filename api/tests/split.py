from common_import import *
@benchmark_registry .register ("split")
class SplitConfig (APIConfig ):
    def __init__ (self ):
        super (SplitConfig ,self ).__init__ ('split')
    def to_pytorch (self ):
        torch_config =super (SplitConfig ,self ).to_pytorch ()
        if isinstance (self .num_or_sections ,int ):
            torch_config .num_or_sections =int (self .x_shape [self .axis ]/
            self .num_or_sections )
        return torch_config
@benchmark_registry .register ("split")
class PaddleSplit (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .split (
        x =x ,num_or_sections =config .num_or_sections ,axis =config .axis )
        if type (result )==list :
            self .fetch_list =result
            config .backward =False
        else :
            self .fetch_list =[result ]
        self .feed_list =[x ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        forward_flop =0
        forward_byte =numel (config .x_shape )*sizeof (config .x_dtype )*2
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("split")
class TorchSplit (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .split (
        tensor =x ,
        split_size_or_sections =config .num_or_sections ,
        dim =config .axis )
        if type (result )==tuple :
            result =list (result )
            self .fetch_list =result
            config .backward =False
        else :
            self .fetch_list =[result ]
        self .feed_list =[x ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("split")
class TFSplit (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .split (
        value =x ,
        num_or_size_splits =config .num_or_sections ,
        axis =config .axis )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])