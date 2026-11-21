from common_import import *
@benchmark_registry .register ("one_hot")
class OneHotConfig (APIConfig ):
    def __init__ (self ):
        super (OneHotConfig ,self ).__init__ ('one_hot')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (OneHotConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec ={"range":[0 ,self .num_classes ]}
@benchmark_registry .register ("one_hot")
class PaddleOneHot (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .one_hot (
        x =x ,num_classes =config .num_classes )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("one_hot")
class TorchOneHot (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .nn .functional .one_hot (
        input =x ,num_classes =config .num_classes )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("one_hot")
class TFOneHot (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .one_hot (
        indices =data ,
        depth =config .num_classes ,
        on_value =None ,
        off_value =None ,
        axis =None ,
        dtype =None )
        self .feed_list =[data ]
        self .fetch_list =[result ]