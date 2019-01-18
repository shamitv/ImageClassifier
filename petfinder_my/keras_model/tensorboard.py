from datetime import datetime
from config.paths import tensorboard_log_dir

from keras.callbacks import TensorBoard

def getTensorboardDirForRun():
    now = datetime.now()
    tstamp = now.strftime("%m_%d_%Y__%H_%M_%S")
    run_log_dir=tensorboard_log_dir+'/'+tstamp
    return run_log_dir


def getTensorboardCallback():
    log_dir=getTensorboardDirForRun()
    return TensorBoard(log_dir=log_dir, update_freq='epoch')