import numpy as np
class Vocab (object ):
    def __init__ (self ,filename =None ,initial_tokens =None ,lower =False ):
        self .id2token ={}
        self .token2id ={}
        self .token_cnt ={}
        self .lower =lower
        self .embed_dim =None
        self .embeddings =None
        self .pad_token ='<пусто>'
        self .unk_token ='<унк>'
        self .initial_tokens =initial_tokens if initial_tokens is not None else []
        self .initial_tokens .extend ([self .pad_token ,self .unk_token ])
        for token in self .initial_tokens :
            self .add (token )
        if filename is not None :
            self .load_from_file (filename )
    def size (self ):
        return len (self .id2token )
    def load_from_file (self ,file_path ):
        for line in open (file_path ,'r'):
            token =line .rstrip ('\n')
            self .add (token )
    def get_id (self ,token ):
        token =token .lower ()if self .lower else token
        try :
            return self .token2id [token ]
        except KeyError :
            return self .token2id [self .unk_token ]
    def get_token (self ,idx ):
        try :
            return self .id2token [idx ]
        except KeyError :
            return self .unk_token
    def add (self ,token ,cnt =1 ):
        token =token .lower ()if self .lower else token
        if token in self .token2id :
            idx =self .token2id [token ]
        else :
            idx =len (self .id2token )
            self .id2token [idx ]=token
            self .token2id [token ]=idx
        if cnt >0 :
            if token in self .token_cnt :
                self .token_cnt [token ]+=cnt
            else :
                self .token_cnt [token ]=cnt
        return idx
    def filter_tokens_by_cnt (self ,min_cnt ):
        filtered_tokens =[token for token in self .token2id if self .token_cnt [token ]>=min_cnt ]
        self .token2id ={}
        self .id2token ={}
        for token in self .initial_tokens :
            self .add (token ,cnt =0 )
        for token in filtered_tokens :
            self .add (token ,cnt =0 )
    def randomly_init_embeddings (self ,embed_dim ):
        self .embed_dim =embed_dim
        self .embeddings =np .random .rand (self .size (),embed_dim )
        for token in [self .pad_token ,self .unk_token ]:
            self .embeddings [self .get_id (token )]=np .zeros ([self .embed_dim ])
    def load_pretrained_embeddings (self ,embedding_path ):
        trained_embeddings ={}
        with open (embedding_path ,'r')as fin :
            for line in fin :
                contents =line .strip ().split ()
                token =contents [0 ].decode ('utf8')
                if token not in self .token2id :
                    continue
                trained_embeddings [token ]=list (map (float ,contents [1 :]))
                if self .embed_dim is None :
                    self .embed_dim =len (contents )-1
        filtered_tokens =trained_embeddings .keys ()
        self .token2id ={}
        self .id2token ={}
        for token in self .initial_tokens :
            self .add (token ,cnt =0 )
        for token in filtered_tokens :
            self .add (token ,cnt =0 )
        self .embeddings =np .zeros ([self .size (),self .embed_dim ])
        for token in self .token2id .keys ():
            if token in trained_embeddings :
                self .embeddings [self .get_id (token )]=trained_embeddings [token ]
    def convert_to_ids (self ,tokens ):
        vec =[self .get_id (label )for label in tokens ]
        return vec
    def recover_from_ids (self ,ids ,stop_id =None ):
        tokens =[]
        for i in ids :
            tokens +=[self .get_token (i )]
            if stop_id is not None and i ==stop_id :
                break
        return tokens