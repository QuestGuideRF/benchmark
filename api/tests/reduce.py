from common_import import *
@benchmark_registry .register ("reduce")
class ReduceConfig (APIConfig ):
    def __init__ (self ):
        super (ReduceConfig ,self ).__init__ ('reduce')
        self .feed_spec ={"range":[-1 ,1 ]}
        self .api_name ='sum'
        self .api_list ={
        'sum':'sum',
        'mean':'mean',
        'max':'max',
        'min':'min',
        'prod':'prod'
        }
    def disabled (self ):
        if self .api_name in ["max","min","prod"
        ]and self .x_dtype =="float16":
            print (
            "Warning:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s.\n"
            %(self .api_name ))
            return True
        return super (ReduceConfig ,self ).disabled ()
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (ReduceConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .axis ==None :
            self .axis =[]
    def to_tensorflow (self ):
        self .api_list ={
        'sum':'reduce_sum',
        'mean':'reduce_mean',
        'max':'reduce_max',
        'min':'reduce_min',
        'prod':'reduce_prod'
        }
        tf_config =super (ReduceConfig ,self ).to_tensorflow ()
        return tf_config
@benchmark_registry .register ("reduce")
class PaddleReduce (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,x =x ,axis =config .axis ,keepdim =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        out_shape =self .fetch_list [0 ].shape
        forward_flop =numel (x_shape )
        forward_byte =(
        numel (x_shape )+numel (out_shape ))*sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("reduce")
class TorchReduce (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        if len (config .axis )==0 :
            result =self .layers (config .api_name ,input =x )
        else :
            result =self .layers (
            config .api_name ,
            input =x ,
            dim =config .axis ,
            keepdim =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("reduce")
class TFReduce (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,
        input_tensor =x ,
        axis =config .axis ,
        keepdims =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])