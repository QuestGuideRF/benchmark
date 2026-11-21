from common_import import *
class PDMaskedSelect (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (
        name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        mask =self .variable (
        name ="mask",shape =config .x_shape ,dtype =config .mask_dtype )
        result =paddle .masked_select (x ,mask )
        self .feed_vars =[x ,mask ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFMaskedSelect (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        tensor =self .variable (
        name ="x",shape =config .x_shape ,dtype =config .x_dtype )
        mask =self .variable (
        name ="mask",shape =config .x_shape ,dtype =config .mask_dtype )
        result =tf .boolean_mask (tensor =tensor ,mask =mask )
        self .feed_list =[tensor ,mask ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[tensor ])
if __name__ =='__main__':
    test_main (PDMaskedSelect (),TFMaskedSelect (),config =APIConfig ("masked_select"))