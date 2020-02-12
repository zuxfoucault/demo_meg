import sys
import yaml


def logger(fileName, logName):
    """ read custom log config"""
    import logging
    import logging.config

    with open('logging.yml') as f:
        LOGGING = yaml.load(f, Loader=yaml.FullLoader)
    LOGGING['handlers']['file']['filename'] = fileName
    logging.config.dictConfig(LOGGING)
    logger = logging.getLogger(logName)
    return logger


def in_jupyter():
    """doc"""
    try:
        cfg = get_ipython().config
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        # print ('NOT Running in ipython notebook env.')
        return False


from contextlib import ContextDecorator
class Timer(ContextDecorator):
    timeit = __import__('timeit')
    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.tstart = self.timeit.default_timer()

    def __exit__(self, type, value, traceback):
        if self.logger:
            if self.name:
                self.logger.info(f'[{self.name}],')
            self.logger.info(f'Elapsed: {(self.timeit.default_timer() - self.tstart)}')
        else:
            if self.name:
                print(f'[{self.name}],')
            print(f'Elapsed: {(self.timeit.default_timer() - self.tstart)}')
# n
