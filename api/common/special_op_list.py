NO_FETCHES_OPS =["feed","null"]
RANDOM_OP_LIST =["bernoulli","dropout"]
OPS_MAP_1TO2 ={
"fill_constant":"full",
"reduce_mean":"mean",
"reduce_prod":"prod",
"reduce_sum":"sum",
}
NO_BACKWARD_OPS =[
"null",
"feed",
"fetch",
"accuracy",
"arange",
"argmax",
"argmin",
"argsort",
"assign",
"cast",
"clip_by_norm",
"cumsum",
"diag",
"equal",
"fill_constant",
"greater_equal",
"greater_than",
"increment",
"isfinite",
"isinf",
"isnan",
"less_equal",
"less_than",
"logical_not",
"logical_and",
"logical_or",
"logical_xor",
"not_equal",
"one_hot",
"scale",
"sequence_mask",
"shape",
"zeros_like",
"add_n",
"all",
"allclose",
"any",
"bernoulli",
"bincount",
"bitwise_and",
"bitwise_not",
"bitwise_or",
"bitwise_xor",
"empty",
"equal_all",
"eye",
"floor_divide",
"full",
"gaussian_random",
"greater",
"isclose",
"less",
"linspace",
"remainder",
"set_value",
"shard_index",
"size",
"truncated_gaussian_random",
"unique",
"viterbi_decode",
"where_index",
"yolo_box",
"lstm",
]
def has_backward (config ):
    if config .framework =="paddle"or not hasattr (config ,"api_list"):
        api_name =config .api_name
    else :
        api_name =None
        for k ,v in config .api_list .items ():
            if v ==config .api_name :
                api_name =k
                break
    return api_name not in NO_BACKWARD_OPS
NO_NEED_ARGS ={
"batch_norm":["moving_mean_name","moving_variance_name"],
"embedding":["is_distributed"]
}
CONTROL_FLOW_OPS =[
"conditional_block","switch","static_rnn","while","while_loop","cond",
"case","ifelse","dynamic_rnn","switch_case"
]
EXCLUDE_OPS =[
"create_tensor","create_parameter","create_global_var",
"autoincreased_step_counter"
]
ALIAS_OP_MAP ={
"arg_max":"argmax",
"arg_min":"argmin",
"nearest_interp":"resize_nearest",
"bilinear_interp":"resize_bilinear",
"depthwise_conv2d":"conv2d",
"sum":"sums",
"hierarchical_sigmoid":"hsigmoid",
"sample_logits":"sampled_softmax_with_cross_entropy"
}
DIFF_IMPLEMENTATION_TF_OPS =["maximum","minimum","argsort"]