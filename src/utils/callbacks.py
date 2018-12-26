import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, LearningRateScheduler

def _lr_schedule(epoch, lr):
    drop_epochs = [40, 80, 120, 160]
    if epoch in drop_epochs:
        lr /= 10
    return lr

def get_callbacks(model_name, checkpoints_dir=None, monitor='val_loss', log_dir=None):

    callbacks = []

    if log_dir is not None:
        directory = os.path.join(log_dir, model_name)
        os.makedirs(directory, exist_ok=True)
        cb = TensorBoard(directory, write_graph=False)
        callbacks.append(cb)

    if checkpoints_dir is not None:
        directory = os.path.join(checkpoints_dir, model_name)
        os.makedirs(directory, exist_ok=True)

        # cb1 = ModelCheckpoint(os.path.join(directory, 'model.h5'), monitor=monitor, verbose=1,
        #                      save_best_only=True, save_weights_only=False,
        #                      mode='auto', period=1)
        # callbacks.append(cb1)

        cb2 = ModelCheckpoint(os.path.join(directory, 'weights-{val_categorical_accuracy:.5}.h5'), monitor=monitor, verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)
        callbacks.append(cb2)

    cb = LearningRateScheduler(_lr_schedule, verbose=1)
    callbacks.append(cb)

    return callbacks

