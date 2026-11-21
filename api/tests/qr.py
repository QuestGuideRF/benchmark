from common_import import *
@benchmark_registry .register ("qr")
class QrConfig (APIConfig ):
    def __init__ (self ):
        super (QrConfig ,self ).__init__ ('qr')
        self .run_torch =False
        print ("[ВНИМАНИЕ]: Pytorch в настоящее время не поддерживает qr.")
@benchmark_registry .register ("qr")
class PaddleQr (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .linalg .qr (x =x ,mode =config .mode )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (list (result ),x )