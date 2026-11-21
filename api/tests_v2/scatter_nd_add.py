from common_import import *
class ScatterNdAddConfig (APIConfig ):
    def __init__ (self ):
        super (ScatterNdAddConfig ,self ).__init__ ('scatter_nd_add')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (ScatterNdAddConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[0 ,int (min (self .input_shape ))]
        },
        {
        "range":[-1 ,1 ]
        }
        ]
class PDScatterNdAdd (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        index =self .variable (
        name ='index',
        shape =config .index_shape ,
        dtype =config .index_dtype ,
        stop_gradient =True )
        updates =self .variable (
        name ='updates',
        shape =config .updates_shape ,
        dtype =config .updates_dtype )
        result =paddle .scatter_nd_add (x =x ,index =index ,updates =updates )
        self .feed_vars =[x ,index ,updates ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,updates ])
class TFScatterNdAdd (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        params =self .variable (
        name ='params',shape =config .input_shape ,dtype =config .input_dtype )
        indices =self .variable (
        name ='indices',shape =config .index_shape ,dtype =config .index_dtype )
        updates =self .variable (
        name ='updates',
        shape =config .updates_shape ,
        dtype =config .updates_dtype )
        result =tf .tensor_scatter_nd_add (
        tensor =params ,indices =indices ,updates =updates )
        self .feed_list =[params ,indices ,updates ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[params ,updates ])
if __name__ =='__main__':
    test_main (PDScatterNdAdd (),TFScatterNdAdd (),config =ScatterNdAddConfig ())