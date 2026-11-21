from common_import import *
class ExpandConfig (APIConfig ):
    def __init__ (self ):
        super (ExpandConfig ,self ).__init__ ('expand')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (ExpandConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .x_shape )==len (self .shape )+1 :
            self .shape =[self .x_shape [0 ]]+self .shape
        assert len (self .x_shape )==len (
        self .shape
        ),"Длина формы должна быть равна рангу входных данных x."
    def to_tensorflow (self ):
        tf_config =super (ExpandConfig ,self ).to_tensorflow ()
        tf_config .multiples =[]
        for i in range (len (self .x_shape )):
            tf_config .multiples .append (self .shape [i ]//self .x_shape [i ])
        return tf_config
class PDExpand (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .expand (x =x ,shape =config .shape )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFExpand (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .tile (input =x ,multiples =config .multiples )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDExpand (),TFExpand (),config =ExpandConfig ())