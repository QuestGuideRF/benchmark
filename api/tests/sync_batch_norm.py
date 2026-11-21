from common_import import *
@benchmark_registry .register ("sync_batch_norm")
class SyncBatchNormConfig (APIConfig ):
    def __init__ (self ):
        super (SyncBatchNormConfig ,self ).__init__ ('sync_batch_norm')
        self .run_torch =False
        self .run_tf =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (SyncBatchNormConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        print ("Предупреждение:\n tensorflow и pytorch не поддерживаются!\n")
        if len (self .x_shape )==4 :
            self .num_channels =self .x_shape [
            1 ]if self .data_format =="NCHW"else self .x_shape [3 ]
        else :
            self .num_channels =self .x_shape [1 ]
@benchmark_registry .register ("sync_batch_norm")
class PaddleSyncBatchNorm (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight_attr =paddle .ParamAttr (
        learning_rate =0.5 ,
        regularizer =paddle .regularizer .L2Decay (1.0 ),
        trainable =True )
        bias_attr =paddle .ParamAttr (
        learning_rate =0.5 ,
        regularizer =paddle .regularizer .L2Decay (1.0 ),
        trainable =True )
        sync_batch_norm =paddle .nn .SyncBatchNorm (
        num_features =config .num_channels ,
        epsilon =config .epsilon ,
        momentum =config .momentum ,
        data_format =config .data_format )
        result =sync_batch_norm (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])