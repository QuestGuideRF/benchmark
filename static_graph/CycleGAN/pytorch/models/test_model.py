from .base_model import BaseModel
from .import networks
class TestModel (BaseModel ):
    @staticmethod
    def modify_commandline_options (parser ,is_train =True ):
        assert not is_train ,'TestModel нельзя использовать во время обучения.'
        parser .set_defaults (dataset_mode ='single')
        parser .add_argument ('--model_suffix',type =str ,default ='',help ='В checkpoints_dir в качестве генератора будет загружен [epoch]_net_G[model_suffix].pth.')
        return parser
    def __init__ (self ,opt ):
        assert (not opt .isTrain )
        BaseModel .__init__ (self ,opt )
        self .loss_names =[]
        self .visual_names =['real_A','fake_B']
        self .model_names =['G'+opt .model_suffix ]
        self .netG =networks .define_G (opt .input_nc ,opt .output_nc ,opt .ngf ,opt .netG ,
        opt .norm ,not opt .no_dropout ,opt .init_type ,opt .init_gain ,self .gpu_ids )
        setattr (self ,'netG'+opt .model_suffix ,self .netG )
    def set_input (self ,input ):
        self .real_A =input ['A'].to (self .device )
        self .image_paths =input ['A_paths']
    def forward (self ):
        self .fake_B =self .netG (self .real_A )
    def optimize_parameters (self ):
        pass