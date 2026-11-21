from common_import import *
@benchmark_registry .register ("kldiv_loss")
class KLDivLossConfig (APIConfig ):
    def __init__ (self ):
        super (KLDivLossConfig ,self ).__init__ ("kldiv_loss")
        self .api_name ='kldiv_loss'
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (KLDivLossConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        input =np .random .rand (*self .input_shape ).astype (self .input_dtype )
        self .input_value =input /input .sum (axis =-1 ,keepdims =True )
        label =np .random .rand (*self .label_shape ).astype (self .label_dtype )
        self .label_value =label /label .sum (axis =-1 ,keepdims =True )
@benchmark_registry .register ("kldiv_loss")
class PaddleKLDivLoss (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',
        shape =config .input_shape ,
        dtype =config .input_dtype ,
        value =config .input_value )
        label =self .variable (
        name ='label',
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        value =config .label_value )
        result =paddle .nn .functional .kl_div (
        input =input ,label =label ,reduction =config .reduction )
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
@benchmark_registry .register ("kldiv_loss")
class TorchKLDivLoss (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',
        shape =config .input_shape ,
        dtype =config .input_dtype ,
        value =config .input_value )
        label =self .variable (
        name ='label',
        shape =config .label_shape ,
        dtype =config .label_dtype ,
        value =config .label_value )
        result =torch .nn .functional .kl_div (
        input =input ,target =label ,reduction =config .reduction )
        self .feed_list =[input ,label ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])