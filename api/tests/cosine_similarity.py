from common_import import *
@benchmark_registry .register ("cosine_similarity")
class PaddleCosineSimilarity (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x1 =self .variable (
        name ='x1',shape =config .x1_shape ,dtype =config .x1_dtype )
        x2 =self .variable (
        name ='x2',shape =config .x2_shape ,dtype =config .x2_dtype )
        result =paddle .nn .functional .cosine_similarity (
        x1 =x1 ,x2 =x2 ,axis =config .axis ,eps =1e-8 )
        self .feed_list =[x1 ,x2 ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x1 ,x2 ])
@benchmark_registry .register ("cosine_similarity")
class TorchCosineSimilarity (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x1 =self .variable (
        name ='x1',shape =config .x1_shape ,dtype =config .x1_dtype )
        x2 =self .variable (
        name ='x2',shape =config .x2_shape ,dtype =config .x2_dtype )
        result =torch .cosine_similarity (
        x1 =x1 ,x2 =x2 ,dim =config .axis ,eps =1e-8 )
        self .feed_list =[x1 ,x2 ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x1 ,x2 ])
@benchmark_registry .register ("cosine_similarity")
class TFCosineSimilarity (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x1 =self .variable (
        name ='x1',shape =config .x1_shape ,dtype =config .x1_dtype )
        x2 =self .variable (
        name ='x2',shape =config .x2_shape ,dtype =config .x2_dtype )
        result =-tf .keras .losses .cosine_similarity (
        y_true =x1 ,y_pred =x2 ,axis =config .axis )
        self .feed_list =[x1 ,x2 ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x1 ,x2 ])