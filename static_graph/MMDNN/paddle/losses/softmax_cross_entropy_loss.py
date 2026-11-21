import static_graph .MMDNN .paddle .layers .paddle_layers as layers
class SoftmaxCrossEntropyLoss (object ):
    def __init__ (self ,conf_dict ):
        pass
    def compute (self ,input ,label ):
        softmax_with_cross_entropy =layers .SoftmaxWithCrossEntropyLayer ()
        reduce_mean =layers .ReduceMeanLayer ()
        cost =softmax_with_cross_entropy .ops (input ,label )
        avg_cost =reduce_mean .ops (cost )
        return avg_cost