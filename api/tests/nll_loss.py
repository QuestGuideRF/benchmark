from common_import import *
@benchmark_registry .register ("nll_loss")
class NllLossConfig (APIConfig ):
    def __init__ (self ):
        super (NllLossConfig ,self ).__init__ ("nll_loss")
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (NllLossConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,self .input_shape [-1 ]]
        }
        ]
@benchmark_registry .register ("nll_loss")
class PaddleNllLoss (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        label =self .variable (
        name ='label',shape =config .label_shape ,dtype =config .label_dtype )
        result =paddle .nn .functional .nll_loss (
        input =input ,
        label =label ,
        ignore_index =config .ignore_index ,
        reduction ='none')
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
@benchmark_registry .register ("nll_loss")
class TorchNllLoss (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        label =self .variable (
        name ='label',shape =config .label_shape ,dtype =config .label_dtype )
        result =torch .nn .functional .nll_loss (
        input =input ,
        target =label ,
        ignore_index =config .ignore_index ,
        reduction ='none')
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])