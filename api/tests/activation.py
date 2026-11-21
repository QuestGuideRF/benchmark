from common_import import *
@benchmark_registry .register ("activation")
class ActivationConfig (APIConfig ):
    def __init__ (self ):
        super (ActivationConfig ,self ).__init__ ('activation')
        self .api_name ='sigmoid'
        self .api_list ={
        'sigmoid':'sigmoid',
        'hardsigmoid':'hardsigmoid',
        'swish':'swish',
        'hardswish':'hardswish',
        'softplus':'softplus',
        'tanhshrink':'tanhshrink',
        'softshrink':'softshrink',
        'softsign':'softsign'
        }
    @property
    def run_tf (self ):
        if self .api_name in [
        "hardsigmoid","hardswish","tanhshrink","softshrink"
        ]:
            print ("-- %s не поддерживается в tf."%self .api_name )
            return False
        return True
    @property
    def run_torch (self ):
        if self .api_name in ["swish"]:
            print ("-- %s не поддерживается в Torch."%self .api_name )
            return False
        return True
@benchmark_registry .register ("activation")
class PaddleActivation (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,module_name ="paddle.nn.functional",x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
    def compute_flop_and_byte (self ,config ):
        x_shape =config .x_shape
        if config .api_name =="sigmoid":
            forward_flop =numel (x_shape )*3
        else :
            forward_flop =None
        forward_byte =numel (x_shape )*2 *sizeof (config .x_dtype )
        if not config .backward :
            return forward_flop ,forward_byte
        else :
            return None ,None
@benchmark_registry .register ("activation")
class TorchActivation (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,module_name ="torch.nn.functional",input =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("activation")
class TFActivation (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])