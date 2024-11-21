from .log import *
from logging import Logger
from typing import Optional, Literal

def print_infomation(msg:str, logger:Optional[Logger], level:Literal['info', 'debug', 'error']='info'):
    if logger is not None:
        match level:
            case 'info':
                logger.info(msg=msg)
            case 'debug':
                logger.debug(msg=msg)
            case 'error':
                logger.error(msg=msg)
    else:
        print(f"[{level}] : {msg}")