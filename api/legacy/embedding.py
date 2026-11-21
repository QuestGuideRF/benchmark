from common_import import *
class EmbeddingConfig (APIConfig ):
    def __init__ (self ):
        super (EmbeddingConfig ,self ).__init__ ('embedding')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (EmbeddingConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,self .size [0 ]]
        },
        {
        "range":[0 ,1 ]
        }
        ]
        if self .is_sparse or self .padding_idx is not None :
            self .run_tf =False
    def disabled (self ):
        if self .dtype =="float16"and not use_gpu ():
            print (
            "Warning:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s на ЦП.\n"
            %(self .api_name ))
            return True
        return super (EmbeddingConfig ,self ).disabled ()
class PDEmbedding (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        table =self .variable (
        name ='table',shape =config .size ,dtype =config .dtype )
        result =fluid .embedding (
        input =input ,
        size =config .size ,
        is_sparse =config .is_sparse ,
        padding_idx =config .padding_idx ,
        param_attr ='table',
        dtype =config .dtype )
        self .feed_vars =[input ,table ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[table ])
class TFEmbedding (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        table =self .variable (
        name ='table',shape =config .size ,dtype =config .dtype )
        result =tf .nn .embedding_lookup (ids =input ,params =table ,max_norm =None )
        self .feed_list =[input ,table ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[table ])
if __name__ =='__main__':
    test_main (PDEmbedding (),TFEmbedding (),config =EmbeddingConfig ())