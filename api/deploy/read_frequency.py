def read_frequency_from_text (op_frequency_path ):
    op_frequency_dict ={}
    with open (op_frequency_path ,"r")as f :
        for line in f .readlines ():
            contents =line .split ()
            if len (contents )!=3 :
                continue
            op_frequency_dict [contents [1 ]]=int (contents [2 ])
    return op_frequency_dict