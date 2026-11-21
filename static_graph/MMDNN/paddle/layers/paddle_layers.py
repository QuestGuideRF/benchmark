import paddle .fluid as fluid
import paddle .fluid .param_attr as attr
class EmbeddingLayer (object ):
    def __init__ (self ,dict_size ,emb_dim ,name ="emb"):
        self .dict_size =dict_size
        self .emb_dim =emb_dim
        self .name =name
    def ops (self ,input ):
        emb =fluid .layers .embedding (input =input ,size =[
        self .dict_size ,self .emb_dim ],is_sparse =True ,
        param_attr =attr .ParamAttr (name =self .name ))
        return emb
class SequencePoolLayer (object ):
    def __init__ (self ,pool_type ):
        self .pool_type =pool_type
    def ops (self ,input ):
        pool =fluid .layers .sequence_pool (
        input =input ,pool_type =self .pool_type )
        return pool
class FCLayer (object ):
    def __init__ (self ,fc_dim ,act ,name ="fc"):
        self .fc_dim =fc_dim
        self .act =act
        self .name =name
    def ops (self ,input ):
        fc =fluid .layers .fc (input =input ,size =self .fc_dim ,param_attr =attr .ParamAttr (
        name ="%s.w"%self .name ),
        bias_attr =attr .ParamAttr (name ="%s.b"%self .name ),act =self .act ,name =self .name )
        return fc
class DynamicGRULayer (object ):
    def __init__ (self ,gru_dim ,name ="dyn_gru"):
        self .gru_dim =gru_dim
        self .name =name
    def ops (self ,input ):
        proj =fluid .layers .fc (input =input ,size =self .gru_dim *3 ,
        param_attr =attr .ParamAttr (name ="%s_fc.w"%self .name ),
        bias_attr =attr .ParamAttr (name ="%s_fc.b"%self .name ))
        gru =fluid .layers .dynamic_gru (input =proj ,size =self .gru_dim ,
        param_attr =attr .ParamAttr (name ="%s.w"%self .name ),
        bias_attr =attr .ParamAttr (name ="%s.b"%self .name ))
        return gru
class DynamicLSTMLayer (object ):
    def __init__ (self ,lstm_dim ,name ="dyn_lstm"):
        self .lstm_dim =lstm_dim
        self .name =name
    def ops (self ,input ):
        proj =fluid .layers .fc (input =input ,size =self .lstm_dim *4 ,
        param_attr =attr .ParamAttr (name ="%s_fc.w"%self .name ),
        bias_attr =attr .ParamAttr (name ="%s_fc.b"%self .name ))
        lstm ,_ =fluid .layers .dynamic_lstm (input =proj ,size =self .lstm_dim *4 ,
        param_attr =attr .ParamAttr (name ="%s.w"%self .name ),
        bias_attr =attr .ParamAttr (name ="%s.b"%self .name ))
        return lstm
class SequenceLastStepLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,input ):
        last =fluid .layers .sequence_last_step (input )
        return last
class SequenceConvPoolLayer (object ):
    def __init__ (self ,filter_size ,num_filters ,name ):
        self .filter_size =filter_size
        self .num_filters =num_filters
        self .name =name
    def ops (self ,input ):
        conv =fluid .nets .sequence_conv_pool (input =input ,filter_size =self .filter_size ,
        num_filters =self .num_filters ,
        param_attr =attr .ParamAttr (name =self .name ),act ="relu")
        return conv
class DataLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,name ,shape ,dtype ,lod_level ):
        data =fluid .layers .data (
        name =name ,shape =shape ,dtype =dtype ,lod_level =lod_level )
        return data
class ConcatLayer (object ):
    def __init__ (self ,axis ):
        self .axis =axis
    def ops (self ,inputs ):
        concat =fluid .layers .concat (inputs ,axis =self .axis )
        return concat
class ReduceMeanLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,input ):
        mean =fluid .layers .reduce_mean (input )
        return mean
class CrossEntropyLayer (object ):
    def __init__ (self ,name ="cross_entropy"):
        pass
    def ops (self ,input ,label ):
        loss =fluid .layers .cross_entropy (input =input ,label =label )
        return loss
class SoftmaxWithCrossEntropyLayer (object ):
    def __init__ (self ,name ="softmax_with_cross_entropy"):
        pass
    def ops (self ,input ,label ):
        loss =fluid .layers .softmax_with_cross_entropy (logits =input ,label =label )
        return loss
class CosSimLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,x ,y ):
        sim =fluid .layers .cos_sim (x ,y )
        return sim
class ElementwiseMaxLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,x ,y ):
        max =fluid .layers .elementwise_max (x ,y )
        return max
class ElementwiseAddLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,x ,y ):
        add =fluid .layers .elementwise_add (x ,y )
        return add
class ElementwiseSubLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,x ,y ):
        sub =fluid .layers .elementwise_sub (x ,y )
        return sub
class ConstantLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,input ,shape ,dtype ,value ):
        constant =fluid .layers .fill_constant_batch_size_like (
        input ,shape ,dtype ,value )
        return constant
class SigmoidLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,input ):
        sigmoid =fluid .layers .sigmoid (input )
        return sigmoid
class SoftsignLayer (object ):
    def __init__ (self ):
        pass
    def ops (self ,input ):
        softsign =fluid .layers .softsign (input )
        return softsign