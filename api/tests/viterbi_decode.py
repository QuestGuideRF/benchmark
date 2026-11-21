from common_import import *
@benchmark_registry .register ("viterbi_decode")
class ViterbiDecodeConfig (APIConfig ):
    def __init__ (self ):
        super (ViterbiDecodeConfig ,self ).__init__ ("viterbi_decode")
        self .run_torch =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (ViterbiDecodeConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,1 ]
        },
        {
        "range":[1 ,self .emission_shape [1 ]+1 ]
        }
        ]
@benchmark_registry .register ("viterbi_decode")
class PaddleViterbiDecode (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        emission =self .variable (
        name ='emission',
        shape =config .emission_shape ,
        dtype =config .emission_dtype )
        transition =self .variable (
        name ='transition',
        shape =config .transition_shape ,
        dtype =config .transition_dtype )
        length =self .variable (
        name ='length',
        shape =config .length_shape ,
        dtype =config .length_dtype )
        scores ,path =paddle .text .viterbi_decode (
        emission ,
        transition ,
        length ,
        include_bos_eos_tag =config .include_bos_eos_tag )
        self .feed_list =[emission ,transition ,length ]
        self .fetch_list =[scores ,path ]