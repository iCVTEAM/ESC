import os
import logging
from datetime import datetime
from torch import distributed as dist


def set_logger(config):
    logging_path = config['logging_path']
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s : %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(os.path.join(logging_path, '{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))),
                            mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    # logging.info('The logger is set.')