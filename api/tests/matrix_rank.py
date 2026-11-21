from common_import import *
@benchmark_registry .register ("matrix_rank")
class MatrixRankConfig (APIConfig ):
    def __init__ (self ):
        super (MatrixRankConfig ,self ).__init__ ("matrix_rank")
        self .feed_spec =[{"range":[-1 ,1 ]}]
@benchmark_registry .register ("matrix_rank")
class PaddleMatrixRank (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .linalg .matrix_rank (
        x =x ,tol =config .tol ,hermitian =config .hermitian )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("matrix_rank")
class TorchMatrixRank (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =torch .linalg .matrix_rank (
        input =x ,tol =config .tol ,hermitian =config .hermitian )
        self .feed_list =[x ]
        self .fetch_list =[result ]