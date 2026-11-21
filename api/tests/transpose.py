from common_import import *
@benchmark_registry .register ("transpose")
class PaddleTranspose (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .transpose (x ,config .perm )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        forward_flop =0
        forward_byte =2 *numel (x_shape )*sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("transpose")
class TorchTranspose (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        if len (config .perm )==2 :
            result =torch .transpose (
            input =x ,dim0 =config .perm [0 ],dim1 =config .perm [1 ]).contiguous ()
        else :
            result =torch .permute (input =x ,dims =config .perm ).contiguous ()
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("transpose")
class TFTranspose (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .transpose (x ,config .perm )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])