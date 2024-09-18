import os, sys
import logging
import requests
from termcolor import colored

from utils.utils import inThread, tryExcept, singleton

logging.basicConfig(level=logging.INFO, format=u'%(filename)s %(levelname)-8s [%(asctime)s] %(message)s')
utils_logger = logging.getLogger(__name__)
utils_logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('logs/utils.log')
fh.setLevel(logging.DEBUG)
utils_logger.addHandler(fh)


def get_logger(name, config):
    '''
Args:
    * name - logger name
    * config -
        possible config keys: 'stdout', 'file'
        there are should be a dict under each keys

        structure for key 'stdout':
            'stdout' : {
                        'level'    : logging.INFO,
                        }

        structure for key 'file':
            'file' :   {
                        'level'    : logging.INFO,
                        'filename' : file.log,
                        }
    * level - main logger level
    '''

    logger = logging.getLogger(name)
    logger.propagate = False

    if 'stdout' in config:
        level = config['stdout']['level']
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(u'%(filename)s [LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if 'file' in config:
        level    = config['file']['level']
        filename = config['file']['filename']
        os.system('echo "{}" >> {}; done'.format('\n'*10 + 'logger inited',  filename))
        handler = logging.FileHandler(filename, "a", encoding=None, delay="true")
        handler.setLevel(level)
        formatter = logging.Formatter(u'%(filename)s [LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_default_stdout_logger(name, level=logging.INFO):
    return get_logger(name, config={'stdout' : {'level' : level}})

def get_default_logger(name, filename, level=logging.INFO, stdout_level=None):
    if stdout_level is None: stdout_level = logging.INFO
    return get_logger(name,
        config={'stdout' : {'level' : stdout_level},
                'file'   : {'level' : level,
                            'filename' : filename}})

def get_default_file_logger(name, filename, level=logging.INFO):
    return get_logger(name, config={'file'   : {'level' : level, 'filename' : filename}})



class ColorizeFilter(logging.Filter):

    color_by_level = {
        logging.DEBUG: 'yellow',
        logging.ERROR: 'red',
        logging.INFO: 'white'
    }

    def filter(self, record):
        record.raw_msg = record.msg
        color = self.color_by_level.get(record.levelno)
        if color:
            record.msg = colored(record.msg, color)
        return True