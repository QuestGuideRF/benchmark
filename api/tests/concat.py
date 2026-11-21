from common_import import *
@benchmark_registry .register ("concat")
class PaddleConcat (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =paddle .concat (x =xs ,axis =config .axis )
        self .feed_list =xs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
    def compute_flop_and_byte (self ,config ):
        forward_flop =0
        forward_byte =0
        for shape in config .x_shape :
            forward_byte +=numel (shape )*sizeof (config .x_dtype [0 ])*2
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("concat")
class TorchConcat (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =torch .cat (tensors =xs ,dim =config .axis )
        self .feed_list =xs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
@benchmark_registry .register ("concat")
class TFConcat (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =tf .concat (values =xs ,axis =config .axis )
        self .feed_list =xs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,xs )