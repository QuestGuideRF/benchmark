from common_import import *
class BernoulliConfig (APIConfig ):
    def __init__ (self ):
        super (BernoulliConfig ,self ).__init__ ('bernoulli')
        self .feed_spec ={"range":[0 ,1 ]}
class PDBernoulli (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .bernoulli (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFBernoulli (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        import tensorflow_probability as tfp
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        b =tfp .distributions .Bernoulli (probs =data )
        result =b .sample ()
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDBernoulli (),TFBernoulli (),config =BernoulliConfig ())