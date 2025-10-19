import argparse
import logging
import yaml
import os


def load_config():
    parser = argparse.ArgumentParser(description='ESegFormer Argparser')
    parser.add_argument('-c', '--config-file', type=str, default='./configs/basic.yaml', help='Path to the config file')
    parser.add_argument('-t', '--task-name', type=str, default=None, help='Name of the task')
    parser.add_argument('-r', '--resume', action='store_true', default=False, help='If True, resume training')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='If True, evaluate only')
    parser.add_argument('-m', '--model-name', type=str, default=None, help='The model name to load')
    parser.add_argument('--debug', action='store_true', default=False, help='If True, Debug Mode, num_workers=0 and batchsize=4')
    '''
    # parser.add_argument('-e', '--eval', action='store_true', default=False, help='If True, evaluate only')
    Evaluation should be created a new python script with a new config yaml file, 
    since train/validation and evaluation are two different process, putting them all in one python file is not a good idea.
    '''
    args = parser.parse_args()
    with open(args.config_file) as file:
        config = yaml.safe_load(file)
    config['config_file'] = args.config_file
    config['task_name'] = args.task_name if args.task_name is not None else config['task_name']
    config['distributed_data_parallel'] = ('SLURM_PROCID' in os.environ and not 'RANK' in os.environ) or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    # If args.resume is set, config['resume'] is True; else if config['resume'] is set, config['resume'] is config['resume']; else False
    # config['eval'] is the same as config['resume']
    config['resume'] = args.resume or config.get('resume', False)
    config['eval'] = args.eval or config.get('eval', False)
    if config['eval']:
        if args.model_name is not None:
            config['model_path'] = os.path.join(config['ckpt']['ckpt_base_path'], config['task_name'], args.model_name)
        else:
            config['model_path'] = os.path.join(config['ckpt']['ckpt_base_path'], config['task_name'], 'model-best.pth.tar')
    if args.debug:
        config['train']['batch_size'] = 4
        config['train']['num_workers'] = 0
        config['val']['batch_size'] = 4
        config['val']['num_workers'] = 0
        config['task_name'] = '{}-debug'.format(config['task_name'])
        config['logger']['logging_path'] = './logs/debug'

    return config