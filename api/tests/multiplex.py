from common_import import *
@benchmark_registry .register ("multiplex")
class MultiplexConfig (APIConfig ):
    def __init__ (self ):
        super (MultiplexConfig ,self ).__init__ ('multiplex')
        self .run_torch =False
        print ("[ВНИМАНИЕ]: Pytorch в настоящее время не поддерживает мультиплексирование.")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (MultiplexConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        index =np .array ([[3 ],[0 ],[1 ],[2 ]])
        self .index_value =index .astype (self .index_dtype )
@benchmark_registry .register ("multiplex")
class PaddleMultiplex (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input =self .variable (
            name ='input'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input )
        index =self .variable (
        name ='index',
        shape =config .index_shape ,
        dtype =config .index_dtype ,
        value =config .index_value )
        result =paddle .multiplex (inputs =inputs ,index =index )
        self .feed_list =[inputs ,index ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,inputs )