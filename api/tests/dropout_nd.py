from common_import import *
import paddle
from paddle import _legacy_C_ops
def dropout_nd (x ,
p =0.5 ,
axis =None ,
training =True ,
mode ="upscale_in_train",
name =None ):
    drop_axes =[axis ]if isinstance (axis ,int )else list (axis )
    seed =None
    mode =('downgrade_in_infer'
    if mode =='downscale_in_infer'else mode )
    out =_legacy_C_ops .dropout_nd (
    x ,
    'dropout_prob',
    p ,
    'is_test',
    not training ,
    'fix_seed',
    seed is not None ,
    'seed',
    seed if seed is not None else 0 ,
    'dropout_implementation',
    mode ,
    'axis',
    drop_axes ,)
    return out
@benchmark_registry .register ("dropout_nd")
class PaddleDropoutNdConfig (APIConfig ):
    def __init__ (self ):
        super (PaddleDropoutNdConfig ,self ).__init__ ('dropout_nd')
        self .run_torch =False
@benchmark_registry .register ("dropout_nd")
class PaddleDropoutNd (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =dropout_nd (
        x =x ,p =config .p ,axis =config .axis ,mode =config .mode )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result [0 ],self .feed_list )