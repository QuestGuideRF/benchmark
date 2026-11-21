from common_import import *
class SyncBatchNormConfig (APIConfig ):
    def __init__ (self ):
        super (SyncBatchNormConfig ,self ).__init__ ('sync_batch_norm')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (SyncBatchNormConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        print ("Warning:\n"  "извините, тензорный поток не поддерживается!\n")
        self .run_tf =False
        if len (self .x_shape )==4 :
            if self .data_format =="NCHW":
                self .num_channels =self .x_shape [1 ]
            else :
                self .num_channels =self .x_shape [3 ]
        else :
            self .num_channels =self .x_shape [1 ]
class PDSyncBatchNorm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        sync_batch_norm =paddle .nn .SyncBatchNorm (
        num_features =config .num_channels ,
        epsilon =config .epsilon ,
        momentum =config .momentum ,
        data_format =config .data_format )
        result =sync_batch_norm (x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDSyncBatchNorm (),config =SyncBatchNormConfig ())