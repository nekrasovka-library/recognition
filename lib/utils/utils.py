import os, io, logging
import numpy as np
import traceback
import argparse

from threading import Thread
from logging.handlers import TimedRotatingFileHandler

logging.basicConfig(level=logging.INFO, format=u'%(filename)s %(levelname)-8s [%(asctime)s] %(message)s')
utils_logger = logging.getLogger(__name__)
utils_logger.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
fh = logging.handlers.TimedRotatingFileHandler('logs/utils.log', when="D", interval=1, backupCount=50)
fh.setLevel(logging.DEBUG)
utils_logger.addHandler(fh)

def str2bool(v):
    if v in [None, 'None', 'none']:
        return None
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean or None value expected.')

def singleton(class_):
    class class_w(class_):
        _instance = None
        def __new__(class_, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w,
                                    class_).__new__(class_)
                class_w._instance._sealed = False
            return class_w._instance
        def __init__(self, *args, **kwargs):
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w

def isCastPossible(newType, value):
    try: 
        newType(value)
        return True
    except ValueError:
        return False


class inThread():
    def __init__(self, daemon=True):
        self.daemon = daemon

    def __call__(self, func):
        def wrapped_f(*args, **kwargs):
            thread = Thread(target=func, daemon=self.daemon, args=args, kwargs=kwargs)
            thread.start()
            return thread
        wrapped_f.__doc__ = func.__doc__
        wrapped_f.__name__ = func.__name__
        return wrapped_f


class tryExcept():
    def __init__(self, logger=utils_logger, verbose=True, writerTelegram=None):
        self.logger = logger
        self.verbose = verbose
        self.writerTelegram = writerTelegram

    def __call__(self, func):
        def wrapped_f(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                if self.verbose:
                    formatted_lines = traceback.format_exc()
                    self.logger.debug('except: {}'.format(formatted_lines))
                    if self.writerTelegram is not None:
                        self.writerTelegram.write('ERROR [OCR_SCALABLE]: {}'.format(formatted_lines))
            return None
        wrapped_f.__doc__ = func.__doc__
        wrapped_f.__name__ = func.__name__
        return wrapped_f


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def get_stats_for_folder(start_path = '.'):
    total_size = 0
    max_timestamp = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                stat = os.stat(fp)
                total_size += stat.st_size
                max_timestamp = max(max_timestamp, stat.st_ctime)
    return total_size, max_timestamp

def get_stats_for_file(path):
    total_size = 0
    max_timestamp = 0
    if not os.path.islink(path):
        stat = os.stat(path)
        total_size += stat.st_size
        max_timestamp = max(max_timestamp, stat.st_ctime)
    return total_size, max_timestamp


def fix_folder_path(path):
    if path is None: return path
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    return path
