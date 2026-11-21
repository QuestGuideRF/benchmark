from common_import import *
class PDCosSim (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .X_shape ,dtype =config .X_dtype )
        y =self .variable (name ='y',shape =config .Y_shape ,dtype =config .Y_dtype )
        result =fluid .layers .cos_sim (X =x ,Y =y )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
class TFCosSim (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .X_shape ,dtype =config .X_dtype )
        y =self .variable (name ='y',shape =config .Y_shape ,dtype =config .Y_dtype )
        result =-tf .keras .losses .cosine_similarity (y_true =x ,y_pred =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
if __name__ =='__main__':
    test_main (PDCosSim (),TFCosSim (),config =APIConfig ("cos_sim"))