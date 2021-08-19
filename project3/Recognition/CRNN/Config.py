from dataclasses import dataclass, replace, field
from typing import List


@dataclass
class HP:

    #re: Resources
    
    experiment_name: str
    exp_dir: str
    #train_data: str
    #valid_data: str
    #langs: List[str]
    #pli: List[int]
    #mode: List[str]
    #task_id: List[int]
    save_path: str = field(init=False)
    select_data: List[int] = field(default_factory=list, init=False)
    batch_ratio: List[int] = field(default_factory=list, init=False)

    #hp: HP
    calc_best_metric_on: str = 'rval' # val datset type [synval|rval] 
    save_iter: int = 1000
    print_iter: int = 100
    character: dict = None
    clova_pretrained_model_path: str = '' 
    pretrained_model_path: str = ''
    spath: str = ''
    Transformation: str = 'None'
    FeatureExtraction: str = 'VGG'
    SequenceModeling: str = 'BiLSTM'
    Prediction: str = 'CTC'
    share: str = 'CNN+LSTM'
    manualSeed: int = 1111
    workers: int = 4
    batch_size: int = 64
    num_iter: int = 300000
    valInterval: int = 1000
    saved_model: str = ''
    FT: bool = False
    adam: bool = True
    rmsprop: bool = False
    total_data_usage_ratio: str = '1.0'
    lr: float = 3e-4
    beta1: float = 0.9
    rho: float = 0.95
    eps: float = 1e-8
    grad_clip: int = 5
    batch_max_length: int = 30
    imgH: int = 32
    imgW: int = 100
    rgb: bool = False
    sensitive: bool = False
    PAD: bool = False
    data_filtering_off: bool = False
    num_fiducial: int = 20
    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256

    def __post_init__(self):
        self.save_path = f'{self.exp_dir}/{self.experiment_name}'
        self.select_data = ['Syn','Real']
        self.batch_ratio = [0,1]


@dataclass
class langConfig:
    lang_name: str
    base_data_path: str # path to folder containing training and validation data
    useReal: bool
    useSyn: bool
    which_real_data: str = 'Real' # if Real/dataset_name specified selects dataset_name if Real uses all datasets under Real 
    which_syn_data: str = 'Syn'
    real_percent: float = 1
    syn_percent: float = 0


@dataclass
class taskConfig:
    task_name: str
    schedule: dict # list of tuples [(lang, num)] => `num` batches of `lang`
    langs: List[langConfig] 
    hp: HP

hp_config = HP(experiment_name='test_loaders_single',exp_dir='exps')
hp_config1 = HP(experiment_name='test_loaders_single',exp_dir='exps',pretrained_model_path = '/workspace/STR/ban_best.pth')



hindi_config = langConfig(lang_name='hindi', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )

ch_config = langConfig(lang_name='ch', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )
korean_config = langConfig(lang_name='korean', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )
japan_config = langConfig(lang_name='japan', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )
arab_config = langConfig(lang_name='arab', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )
ban_config = langConfig(lang_name='ban', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )
latin_config = langConfig(lang_name='latin', base_data_path='/workspace/STR/data',
                useSyn=False, useReal=True
            )          

#task_kan = taskConfig(task_name='kan',langs=[kan_config],schedule=[('kan',1)],hp=hp_config) 
task_hi = taskConfig(task_name='hindi',langs=[hindi_config],schedule=[('hindi',1)],hp=hp_config)
task_korean = taskConfig(task_name='korean',langs=[korean_config],schedule=[('korean',1)],hp=hp_config)
#task_kan_hi = taskConfig(task_name='kan_hin',langs=[kan_config,hin_config3],schedule=[('kan',1),('hin',1)],hp=hp_config)


task_multi = taskConfig(task_name='multilanguage',

langs = [hindi_config,ch_config,arab_config,ban_config,latin_config,korean_config,japan_config],
schedule=[('hindi',1),('ch',1),('arab',1),('ban',1),('latin',1),('korean',1),('japan',1)],
hp=hp_config1)

task_hi_ban = taskConfig(task_name = 'hindi_ban',langs = [hindi_config,ban_config],schedule=[('hindi',1),('ban',1)],hp=hp_config)

task_ch = taskConfig(task_name = 'ch',langs = [ch_config],schedule = [('ch',1)],hp=hp_config1)

task_ban = taskConfig(task_name = 'ban',langs = [ban_config],schedule=[('ban',1)],hp=hp_config1)

