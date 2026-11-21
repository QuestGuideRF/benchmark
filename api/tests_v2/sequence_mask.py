from common_import import *
class SequenceMaskConfig (APIConfig ):
    def __init__ (self ):
        super (SequenceMaskConfig ,self ).__init__ ('sequence_mask')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (SequenceMaskConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if hasattr (self ,"maxlen_shape"):
            self .maxlen =3
class PDSequenceMask (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        if hasattr (config ,"maxlen_shape"):
            maxlen =paddle .full (
            shape =config .maxlen_shape ,
            fill_value =config .maxlen ,
            dtype =config .maxlen_dtype )
        else :
            maxlen =config .maxlen
        result =paddle .fluid .layers .sequence_mask (
        x =data ,maxlen =maxlen ,dtype =config .dtype )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFSequenceMask (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .sequence_mask (
        lengths =data ,
        maxlen =config .maxlen ,
        dtype =tf .as_dtype (config .dtype ))
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDSequenceMask (),TFSequenceMask (),config =SequenceMaskConfig ())