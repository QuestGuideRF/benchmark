from common_import import *
@benchmark_registry .register ("stack")
class PaddleStack (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =paddle .stack (x =xs ,axis =config .axis )
        self .feed_list =xs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
    def compute_flop_and_byte (self ,config ):
        num_xs =len (config .x_shape )
        forward_flop =0
        forward_byte =num_xs *numel (config .x_shape [0 ])*sizeof (
        config .x_dtype [0 ])*2
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("stack")
class TorchStack (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        values =[]
        for i in range (len (config .x_shape )):
            value_i =self .variable (
            name ='value_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            values .append (value_i )
        result =torch .stack (tensors =values ,dim =config .axis )
        self .feed_list =values
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,values )
@benchmark_registry .register ("stack")
class TFStack (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        values =[]
        for i in range (len (config .x_shape )):
            value_i =self .variable (
            name ='value_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            values .append (value_i )
        result =tf .stack (values =values ,axis =config .axis )
        self .feed_list =values
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,values )