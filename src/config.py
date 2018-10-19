from os.path import join

### Pathes
PROJECT_DIR = '/home/user/project'

# top level folders
DATA_DIR = join(PROJECT_DIR, 'data')
CHECKPOINTS_DIR = join(PROJECT_DIR, 'checkpoints')
LOGS_DIR = join(PROJECT_DIR, 'logs')

# second level folders
RAW_DATA_DIR = join(DATA_DIR, 'raw_data')
DATASET_DIR = join(DATA_DIR, 'dataset')
SUBMISSION_DIR = join(DATA_DIR, 'submissions')
PREDICTIONS_DIR = join(DATA_DIR, 'predictions')

TRAIN_DIR = join(DATASET_DIR, 'train')
VALID_DIR = join(DATASET_DIR, 'val')

VAL_GT = join(DATASET_DIR, 'ILSVRC2012_val_gt.txt')
MAP_CLS = join(DATASET_DIR, 'imagenet_class_index.json')