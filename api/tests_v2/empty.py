from common_import import *
class EmptyConfig (APIConfig ):
    def __init__ (self ):
        super (EmptyConfig ,self ).__init__ ('empty')
        self .run_tf =False
class PDEmpty (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        result =paddle .empty (shape =config .shape ,dtype =config .dtype )
        self .feed_vars =[]
        self .fetch_vars =[result ]
if __name__ =='__main__':
    test_main (PDEmpty (),config =EmptyConfig ())