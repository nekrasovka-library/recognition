# GD

import os, sys
from tqdm import tqdm
import logging
from utils.tqdm_utils import TqdmToLogger
from utils.logger_utils import get_default_stdout_logger

class BaseModel(object):
    name             = 'BaseModel'
    _base_init_args  = {'logger', 'tqdm_type'}

    def __init__(self, logger, tqdm_type):
        self.set_logger(logger)
        self.set_tqdm(tqdm_type)

    def set_n_jobs(self, n_jobs):
        self.n_jobs = n_jobs

    def set_logger(self, logger):
        if logger is None:
            logger = get_default_stdout_logger(self.name)
        self.logger = logger

    def set_tqdm(self, tqdm_type, **kw):
        if tqdm_type == 'default':
            self.tqdm = lambda iterable, **kw: tqdm(iterable, **kw)
        elif tqdm_type == 'list':
            self.tqdm = lambda iterable : iterable
        elif tqdm_type == 'logger':
            tqdm_out  = TqdmToLogger(self.logger, level=logging.INFO)
            if 'miniters' not in kw: kw['miniters']=10
            self.tqdm = lambda iterable, **kw: tqdm(iterable, file=tqdm_out, **kw)
        else:
            raise NotImplementedError('Unknown tqdm_type: {}'.format(tqdm_type))

    def run(self):
        raise NotImplementedError()
