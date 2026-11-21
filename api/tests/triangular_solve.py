from common_import import *
@benchmark_registry .register ("triangular_solve")
class TriangularSolveConfig (APIConfig ):
    def __init__ (self ):
        super (TriangularSolveConfig ,self ).__init__ ('triangular_solve')
        self .run_torch =False
@benchmark_registry .register ("triangular_solve")
class PaddleTriangularSolve (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .linalg .triangular_solve (
        x =x ,
        y =y ,
        upper =config .upper ,
        transpose =config .transpose ,
        unitriangular =config .unitriangular )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])