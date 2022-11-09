
import numpy as np
import os
import matplotlib.pyplot as plt


import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


from ml_utils.dl_architectures import *
import json
from keras.models import model_from_json

from tensorflow.python.platform import tf_logging as logging
import re
from keras.distribute import distributed_file_utils
from keras.utils import tf_utils
from keras.utils import io_utils



class lastModelCheckpoint(keras.callbacks.Callback):
  """Callback to save the Keras model or model weights at some frequency.
  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.
  A few options this callback provides include:
  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.
  Note: If you get `WARNING:tensorflow:Can save best model only with <name>
  available, skipping` see the description of the `monitor` argument for
  details on how to get this right.
  Example:
  ```python
  model.compile(loss=..., optimizer=...,
                metrics=['accuracy'])
  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)
  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```
  Args:
      filepath: string or `PathLike`, path to save the model file. e.g.
        filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename. The directory of the filepath should not be reused by
        any other callbacks to avoid conflicts.
      monitor: The metric name to monitor. Typically the metrics are set by the
        `Model.compile` method. Note:
        * Prefix the name with `"val_`" to monitor validation metrics.
        * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
        * If you specify metrics as strings, like `"accuracy"`, pass the same
          string (with or without the `"val_"` prefix).
        * If you pass `metrics.Metric` objects, `monitor` should be set to
          `metric.name`
        * If you're not sure about the metric names you can check the contents
          of the `history.history` dictionary returned by
          `history = model.fit()`
        * Multi-output models set additional prefixes on the metric names.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
        displays messages when the callback takes an action.
      save_best_only: if `save_best_only=True`, it only saves when the model
        is considered the "best" and the latest best model according to the
        quantity monitored will not be overwritten. If `filepath` doesn't
        contain formatting options like `{epoch}` then `filepath` will be
        overwritten by each new better model.
      mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
        monitored are 'acc' or start with 'fmeasure' and are set to `min` for
        the rest of the quantities.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      initial_value_threshold: Floating point initial "best" value of the metric
        to be monitored. Only applies if `save_best_value=True`. Only overwrites
        the model weights already saved if the performance of current
        model is better than this value.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               options=None,
               initial_value_threshold=None,
               epoch_last_save = 0, 
               **kwargs):
    super(lastModelCheckpoint, self).__init__()
    self._supports_tf_logs = True
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = io_utils.path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.save_freq = save_freq
    self.epochs_since_last_save = 0
    self.epoch_last_save = epoch_last_save
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0
    self.best = initial_value_threshold

    if save_weights_only:
      if options is None or isinstance(
          options, tf.train.CheckpointOptions):
        self._options = options or tf.train.CheckpointOptions()
      else:
        raise TypeError(
            'If save_weights_only is True, then `options` must be '
            f'either None or a tf.train.CheckpointOptions. Got {options}.')
    else:
      if options is None or isinstance(options, tf.saved_model.SaveOptions):
        self._options = options or tf.saved_model.SaveOptions()
      else:
        raise TypeError(
            'If save_weights_only is False, then `options` must be '
            f'either None or a tf.saved_model.SaveOptions. Got {options}.')

    # Deprecated field `load_weights_on_restart` is for loading the checkpoint
    # file from `filepath` at the start of `model.fit()`
    # TODO(rchao): Remove the arg during next breaking release.
    if 'load_weights_on_restart' in kwargs:
      self.load_weights_on_restart = kwargs['load_weights_on_restart']
      logging.warning('`load_weights_on_restart` argument is deprecated. '
                      'Please use `model.load_weights()` for loading weights '
                      'before the start of `model.fit()`.')
    else:
      self.load_weights_on_restart = False

    # Deprecated field `period` is for the number of epochs between which
    # the model is saved.
    if 'period' in kwargs:
      self.period = kwargs['period']
      logging.warning('`period` argument is deprecated. Please use `save_freq` '
                      'to specify the frequency in number of batches seen.')
    else:
      self.period = 1

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      if self.best is None:
        self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      if self.best is None:
        self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        if self.best is None:
          self.best = -np.Inf
      else:
        self.monitor_op = np.less
        if self.best is None:
          self.best = np.Inf

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError(
          f'Unrecognized save_freq: {self.save_freq}. '
          'Expected save_freq are "epoch" or integer')

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def on_train_begin(self, logs=None):
    if self.load_weights_on_restart:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError(
              f'Error loading file from {filepath_to_load}. Reason: {e}')

  def _implements_train_batch_hooks(self):
    # Only call batch hooks when saving on batch
    return self.save_freq != 'epoch'

  def on_train_batch_end(self, batch, logs=None):
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      self._save_model(epoch=epoch, batch=None, logs=logs)

  def _should_save_on_batch(self, batch):
    """Handles batch-level saving logic, supports steps_per_execution."""
    if self.save_freq == 'epoch':
      return False

    if batch <= self._last_batch_seen:  # New epoch.
      add_batches = batch + 1  # batches are zero-indexed.
    else:
      add_batches = batch - self._last_batch_seen
    self._batches_seen_since_last_saving += add_batches
    self._last_batch_seen = batch

    if self._batches_seen_since_last_saving >= self.save_freq:
      self._batches_seen_since_last_saving = 0
      return True
    return False

  def _save_model(self, epoch, batch, logs):
    """Saves the model.
    Args:
        epoch: the epoch this iteration is in.
        batch: the batch this iteration is in. `None` if the `save_freq`
          is set to `epoch`.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, batch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                io_utils.print_msg(
                    f'\nEpoch {epoch + 1}: {self.monitor} improved '
                    f'from {self.best:.5f} to {current:.5f}, '
                    f'saving model to {filepath}')
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                    filepath, overwrite=True, options=self._options)
              else:
                self.model.save(filepath, overwrite=True, options=self._options)
            else:
              if self.verbose > 0:
                io_utils.print_msg(
                    f'\nEpoch {epoch + 1}: '
                    f'{self.monitor} did not improve from {self.best:.5f}')
        else:
          if self.verbose > 0:
            io_utils.print_msg(
                f'\nEpoch {epoch + 1}: saving model to {filepath}')
          if self.save_weights_only:
            self.model.save_weights(
                filepath, overwrite=True, options=self._options)
          else:
            self.model.save(filepath, overwrite=True, options=self._options)

        self._maybe_remove_file()
      except IsADirectoryError as e:  # h5py 3.x
        raise IOError('Please specify a non-directory filepath for '
                      'ModelCheckpoint. Filepath used is an existing '
                      f'directory: {filepath}')
      except IOError as e:  # h5py 2.x
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        f'directory: f{filepath}')
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, batch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}`,`{batch:02d}`
      # and `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      if batch is None or 'batch' in logs:
        file_path = self.filepath.format(epoch=(epoch + 1 + int(self.epoch_last_save)), **logs)
      else:
        file_path = self.filepath.format(
            epoch=(epoch + 1 + int(self.epoch_last_save)) , batch=batch + 1, **logs)
    except KeyError as e:
      raise KeyError(
          f'Failed to format this callback filepath: "{self.filepath}". '
          f'Reason: {e}')
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return tf.io.gfile.exists(filepath)
    tf_saved_model_exists = tf.io.gfile.exists(filepath)
    tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.
    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.
    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.
    Modified time of a file is obtained with `os.path.getmtime()`.
    This utility function is best demonstrated via an example:
    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```
    Args:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.
    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if tf.io.gfile.exists(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name


def minmax_scale(data, minval = None, maxval = None):
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    return (data - minval) / ((maxval - minval))

def descaledminamax(scaleddata, minval, maxval):
    #(data - minval) / ((maxval - minval)) =de
    data = (scaleddata*(maxval-minval)) + minval
    return data

def descaledstandard(scaleddata, meanval, sdtval):
    #(data - minval) / ((maxval - minval)) =de
    data = (scaleddata*(sdtval)) + meanval
    return data

def rmse_m(y_true, y_pred):
    error = y_true - y_pred
    sqr_error = K.square(error)
    mean_sqr_error = K.mean(sqr_error)
    sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
    return sqrt_mean_sqr_error


def built_dlmodel(inputsshape,dlmodelname, noutputs = 1, 
                   final_activation = "linear", initialfilters = 64, 
                   finaldense = 512,finaldrop = 0.4):
                   
    tf.keras.backend.clear_session()
    model = None
    if dlmodelname == "conv3d":
        depth, width, height, channels = inputsshape
        inputm = set_Conv3dmodel(width=width, height=height, depth=depth, 
                                 channels = channels,initfilters = initialfilters)
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm.output)
        inputsmodel = inputm.input

    if dlmodelname == "conv3dlstm":
        # dates, channels, pixels_x, pixels_y
        depth, width, height, channels = inputsshape
        inputm = ConvLSTM_Model(frames=depth, channels=channels, 
                                width=width, height=height,initfilters = initialfilters)
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm.output)
        inputsmodel = inputm.input

    if dlmodelname == "alexnet3d":
        depth, width, height, channels = inputsshape
        
        inputm = alexnet3d(width=width, height=height, depth=depth, channels = channels)
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm.output)
        inputsmodel = inputm.input


    if dlmodelname == "resnet3d":
        depth, width, height, channels = inputsshape
        inputm = ResNet34_3d(width=width, height=height, depth=depth, channels = channels)
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm.output)
        inputsmodel = inputm.input

    if dlmodelname == "conv_conv3dlstm":
        depth, width, height, channels = inputsshape
        inputm = set_Conv3d_ConvLstm_model(width=width, height=height, depth=depth, channels = channels)
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm.output)
        inputsmodel = inputm.input


    if dlmodelname == "conv3d_conc":

        depth, width, height, channels = inputsshape[0]
        modelimg = set_Conv3dmodel(width=width, height=height, depth=depth, 
                                        channels = channels,initfilters = initialfilters)

        depth, width, height, channels = inputsshape[1]
        modelzp = set_Conv3dmodel(width=width, height=height, depth=depth, 
                                        channels = channels,initfilters = initialfilters)
        
        inputm = keras.layers.concatenate([modelimg.output, modelzp.output])
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm)
        inputsmodel = [modelimg.input,
                           modelzp.input]
    
    if dlmodelname == "conv3dlstm_conc":

        depth, width, height, channels = inputsshape[0]
        modelimg = ConvLSTM_Model_zprof(frames=depth, channels=channels, 
                                width=width, height=height,initfilters = initialfilters)

        depth, width, height, channels = inputsshape[1]
        modelzp = ConvLSTM_Model_zprof(frames=depth, channels=channels, 
                                width=width, height=height,initfilters = initialfilters)
        
        inputm = keras.layers.concatenate([modelimg.output, modelzp.output])
        x = keras.layers.Dense(finaldense, activation=final_activation)(inputm)
        inputsmodel = [modelimg.input,
                           modelzp.input]
    
    if dlmodelname == "conv3d_concdeco":
      
      depth, width, height, channels = inputsshape[0]
      modelimg = set_Conv3dmodel_block(width=width, height=height, depth=depth, 
                                      channels = channels)

      depth, width, height, channels = inputsshape[1]
      modelzp = set_Conv3dmodel_block(width=width, height=height, depth=depth, 
                                      channels = channels)
      
      inputm = layers.concatenate([modelimg.output, modelzp.output])

      x  = keras.layers.BatchNormalization()(inputm)

      x  = layers.Conv2D(64, (5, 5), strides=2, activation="relu", padding="same")(x)
      x  = keras.layers.BatchNormalization()(x)

      x  = layers.Conv2D(32, (5, 5), strides=2, activation="relu", padding="same")(x)
      x  = keras.layers.BatchNormalization()(x)
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Flatten()(x)
      x = layers.Dropout(0.3)(x)
      x = keras.layers.Dense(finaldense, activation=final_activation)(x)
      inputsmodel = [modelimg.input,
                           modelzp.input]

        
    if dlmodelname == "conv3dlstm_concdeco":

      depth, width, height, channels = inputsshape[0]
      modelimg = ConvLSTM_Model_block(frames=depth, channels=channels, 
                              width=width, height=height,initfilters = initialfilters)

      depth, width, height, channels = inputsshape[1]
      modelzp = ConvLSTM_Model_block(frames=depth, channels=channels, 
                              width=width, height=height,initfilters = initialfilters)

      inputm = layers.concatenate([modelimg.output, modelzp.output])

      x  = keras.layers.BatchNormalization()(inputm)

      x  = layers.Conv2D(64, (5, 5), strides=2, activation="relu", padding="same")(x)
      x  = keras.layers.BatchNormalization()(x)

      x  = layers.Conv2D(32, (5, 5), strides=2, activation="relu", padding="same")(x)
      x  = keras.layers.BatchNormalization()(x)
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Flatten()(x)
      x = layers.Dropout(0.3)(x)              
      x = keras.layers.Dense(finaldense, activation=final_activation, 
                             kernel_regularizer=tf.keras.regularizers.L1(0.01))(x)
      inputsmodel = [modelimg.input,
                           modelzp.input]


    x = keras.layers.Dropout(finaldrop)(x)
    mlastlayer = layers.Dense(noutputs)(x)
    model= keras.Model(inputs=inputsmodel, outputs=mlastlayer, name = dlmodelname)

    print(final_activation,finaldrop,noutputs)
    if model is None:
        raise ValueError("please chen the available model's configuration {}".format(["conv3d","alexnet3d","conv3dlstm"]))
    print("{} model configurated".format(dlmodelname))
    return model

def findepochnumber(weigthfiles, index, strepochref = 'epoch', stopepochstr ='-loss' ):
  flindex = [i[:-6] for i in weigthfiles if i.endswith(index)]  
  
  return np.array([int(i[(i.index(strepochref)+5):(i.index(stopepochstr))]) for i in flindex]), flindex



class dlModel:
    
    def _checkdata(self, input):
        
        modelshape = np.array(self.model.input.shape[1:])
        inputshape = np.array(input[0].shape)

        if not np.array_equal(modelshape,inputshape):
            raise ValueError('the shapes are different')
    
    def restore_weights(self):

      tfiles = os.listdir(self.checkpoint_path)
      print(self._load_last)

      if self._load_last:
        flindex = [i[:-6] for i in tfiles if i.startswith("testfinal") and i.endswith("index")]
        print(flindex)
        if len(flindex)>0:
          epochnumber = np.array([int(i[10:]) for i in flindex])

        else:
          epochnumber, flindex = findepochnumber(tfiles, index="_run.index")

      else:
        epochnumber, flindex = findepochnumber(tfiles, index="_run.index")
      
      if len(epochnumber)>0:
          bestmodel = flindex[np.where(epochnumber == max(epochnumber))[0][0]]
          self.model.load_weights(os.path.join(self.checkpoint_path, bestmodel))
          self.bestmodel = bestmodel
          epochnumber = np.array(epochnumber)
          epochnumber[::-1].sort()
          self._last_epoch = epochnumber[0]

          print(self._last_epoch)
          print("checkpoint load {}".format(os.path.join(self.checkpoint_path, bestmodel)))
      else:
          print("it was not possible to load weights **********")
        


    def set_model_config(self, inputshape):


        drop_perc = self.parameters['drop_perc'] if 'drop_perc' in self.parameters else 0.4
        dense_units = self.parameters['dense_units'] if 'dense_units' in self.parameters else 512
        final_activation = self.parameters['final_activation'] if 'final_activation' in self.parameters else 'linear'
        initfilters = self.parameters['initfilters'] if 'initfilters' in self.parameters else 64
        print(self.parameters)
        if self.architecture.endswith('json'):
            with open(self.architecture, 'r') as json_file:
                architecture = json.load(json_file)
            self.model = model_from_json(json.dumps(architecture))
            
        else:
            self.model = built_dlmodel(inputshape, self.architecture, 
                                        finaldense=dense_units,
                                        initialfilters = initfilters,
                                        finaldrop=drop_perc,
                                        final_activation=final_activation)

        if self.checkpoint_path is not None:
            if os.path.exists(self.checkpoint_path):
                self.restore_weights()

        else:
            print("model uploaded from scratch")     

        lossfun = self.parameters['lossfun'] if 'lossfun' in self.parameters else 'mae'
        lr = self.parameters['lr'] if 'lr' in self.parameters else 0.0001
        momentum = self.parameters['momentum'] if 'momentum' in self.parameters else 0.9

        self.model.compile(loss=lossfun,
                    optimizer=tf.keras.optimizers.SGD(lr, momentum = momentum),
                    metrics=rmse_m)

    
    def training(self, trainingdada,validationdata, 
                callbacklist, epochs = 10):
        
        tr_batch = self.parameters['tr_batch'] if 'tr_batch' in self.parameters else 20
        eval_batch = self.parameters['eval_batch'] if 'eval_batch' in self.parameters else 20

        inputtraining = trainingdada[0]
        self._checkdata(inputtraining)
        targettraining = trainingdada[1]

        inputvalidation = validationdata[0]
        targetvalidation = validationdata[1]
        
        self.model.fit(
            inputtraining,
            targettraining,
            epochs=epochs,
            shuffle=True,
            verbose=1,
            batch_size = tr_batch,
            callbacks=callbacklist,
            validation_data = (inputvalidation, targetvalidation),
            validation_batch_size=eval_batch
        )
        
    
    def evaluate(self, input, scaley = None, plot = True):

        yprediction = self.model.predict(input[0])
        if scaley is not None:
            yprediction = descaledminamax(yprediction.flatten(), scaley[0], scaley[1])

        #rmse = rmse_m(yprediction,input[1])
        #print("RMSE: {}".format(rmse))

        #if plot:
        #    plt.scatter(yprediction,input[1])
        #    plt.title('RMSE: {:.3f}'.format(rmse))
        
        return yprediction
            

    def evaluation_metrics(self):
        if self.targetscale is not None:
            targetvalue =  descaledminamax(self.targettest, self.targetscale['min'], self.targetscale['max'])
        else:
            targetvalue = self.targettest
        rmse = rmse_m(self.yprediction,targetvalue)
        r2 = rmse_m(self.yprediction,self.targettest)

        return [rmse, r2]
        
    def __init__(self,
                 inputshape,
                 architecture = None,
                 checkpoint_path = None,
                 parameters = None,
                 load_lastweigths = False
                 ):
        self.bestmodel = None
        #self.targetscale = targetvariable["scaler"] if "scaler" in targetvariable else None
        self.architecture = architecture
        self.checkpoint_path = checkpoint_path
        self._load_last = load_lastweigths
        self._last_epoch = None
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = {'lossfun':'mae',
                               'lr': 0.0001,
                               'momentum': 0.9,
                               'drop_perc': 0.2,
                               'dense_units': 512,
                               'initfilters':64,
                               'tr_batch': 20,
                               'eval_batch': 32,
                               'final_activation': 'linear'}
        

        ## set dl model
        self.set_model_config(inputshape)

        
