from .base_options import BaseOptions
class TestOptions (BaseOptions ):
    def initialize (self ,parser ):
        parser =BaseOptions .initialize (self ,parser )
        parser .add_argument ('--ntest',type =int ,default =float ("inf"),help ='Количество тестовых примеров.')
        parser .add_argument ('--results_dir',type =str ,default ='./results/',help ='сохраняет результаты здесь.')
        parser .add_argument ('--aspect_ratio',type =float ,default =1.0 ,help ='соотношение сторон результирующих изображений')
        parser .add_argument ('--phase',type =str ,default ='test',help ='поезд, вал, тест и т. д.')
        parser .add_argument ('--eval',action ='store_true',help ='используйте режим оценки во время тестирования.')
        parser .add_argument ('--num_test',type =int ,default =50 ,help ='сколько тестовых изображений запустить')
        parser .set_defaults (model ='test')
        parser .set_defaults (load_size =parser .get_default ('crop_size'))
        self .isTrain =False
        return parser