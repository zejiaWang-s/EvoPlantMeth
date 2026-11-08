from __future__ import division, print_function

from collections import OrderedDict
import os
from time import time
import numpy as np
import six

from tensorflow.keras.callbacks import Callback
from .utils import format_table

class PerformanceLogger(Callback):
    def __init__(self, metrics=['loss', 'acc'], log_freq=0.1, precision=4, callbacks=[], verbose=bool, logger=print):
        super(PerformanceLogger, self).__init__()
        self.metrics = metrics
        self.log_freq = log_freq
        self.precision = precision
        self.callbacks = callbacks
        self.verbose = verbose
        self.logger = logger
        self._line = '=' * 100
        self.epoch_logs = None
        self.val_epoch_logs = None
        self.batch_logs = []

    def _log(self, x):
        if self.logger: self.logger(x)

    def _init_logs(self, logs, train=True):
        logs = list(logs)
        logs = [log for log in logs if not log.startswith('val_')] if train else [log[4:] for log in logs if log.startswith('val_')]
        metrics = OrderedDict()
        
        for name in self.metrics:
            if name in logs: metrics[name] = [name]
            output_logs = [log for log in logs if log.endswith('_' + name)]
            if output_logs:
                metrics.setdefault(name, [name]).extend(output_logs)

        logs_dict = OrderedDict()
        for mean_name in metrics: logs_dict[mean_name] = []
        for names in six.itervalues(metrics):
            for name in names: logs_dict[name] = []
            
        return metrics, logs_dict

    def _update_means(self, logs, metrics):
        for mean_name, names in six.iteritems(metrics):
            if logs[mean_name][-1] is not None: continue
            valid_vals = [logs[n][-1] for n in names if n in logs and logs[n][-1] is not None and not np.isnan(logs[n][-1])]
            logs[mean_name][-1] = sum(valid_vals) / len(valid_vals) if valid_vals else np.nan

    def on_train_begin(self, logs={}):
        self._time_start = time()
        if hasattr(self, 'params') and 'epochs' in self.params:
            self._log('Epochs: %d' % self.params['epochs'])

    def on_train_end(self, logs={}):
        self._log(self._line)

    def on_epoch_begin(self, epoch, logs={}):
        self._log(self._line)
        epochs = self.params.get('epochs')
        self._log(f"Epoch {epoch + 1}/{epochs}" if epochs else f"Epoch {epoch + 1}")
        self._log(self._line)
        
        self._step = 0
        self._steps = self.params.get('steps')
        self._log_freq = int(np.ceil(self.log_freq * self._steps)) if self._steps else 1
        self._batch_logs = None
        self._totals = None

    def on_epoch_end(self, epoch, logs={}):
        if self._batch_logs: self.batch_logs.append(self._batch_logs)

        if not self.epoch_logs:
            self._epoch_metrics, self.epoch_logs = self._init_logs(logs)
            self._val_epoch_metrics, self.val_epoch_logs = self._init_logs(logs, False)

        for metric, metric_logs in six.iteritems(self.epoch_logs):
            metric_logs.append(logs.get(metric))
        self._update_means(self.epoch_logs, self._epoch_metrics)

        for metric, metric_logs in six.iteritems(self.val_epoch_logs):
            metric_logs.append(logs.get('val_' + metric))
        self._update_means(self.val_epoch_logs, self._val_epoch_metrics)

        table = OrderedDict({'split': ['train']})
        for mean_name in self._epoch_metrics: table[mean_name] = []
        if self.verbose:
            for names in six.itervalues(self._epoch_metrics):
                for name in names: table[name] = []
        for name, logs_ in six.iteritems(self.epoch_logs):
            if name in table: table[name].append(logs_[-1])
            
        if self.val_epoch_logs and any(v is not None for v in self.val_epoch_logs.values()):
            table['split'].append('val')
            for name, logs_ in six.iteritems(self.val_epoch_logs):
                if name in table: table[name].append(logs_[-1])
                
        self._log('')
        self._log(format_table(table, precision=self.precision))

        for callback in self.callbacks:
            callback(epoch, self.epoch_logs, self.val_epoch_logs)

    def on_batch_end(self, batch, logs={}):
        self._step += 1
        batch_size = logs.get('size', logs.get('batch', 0))

        if self._steps is None:
            self._steps = self.params.get('steps')
            self._log_freq = int(np.ceil(self.log_freq * self._steps)) if self._steps else 1

        if not self._batch_logs:
            self._batch_metrics, self._batch_logs = self._init_logs(logs.keys())
            self._totals, self._nb_totals = OrderedDict(), OrderedDict()
            for name in self._batch_logs:
                if name in logs: self._totals[name], self._nb_totals[name] = 0, 0

        for name, value in six.iteritems(logs):
            if np.isnan(value): continue
            if name in self._totals:
                self._totals[name] += value * batch_size
                self._nb_totals[name] += batch_size

        for name in self._batch_logs:
            if name in self._totals:
                self._batch_logs[name].append(self._totals[name] / self._nb_totals[name] if self._nb_totals[name] else np.nan)
            else:
                self._batch_logs[name].append(None)
                
        self._update_means(self._batch_logs, self._batch_metrics)
        
        do_log = (self._step % self._log_freq == 0) or self._step == 1 or \
                 (self._step == self._steps if self._steps else self._step % 100 == 0)

        if do_log:
            table = OrderedDict()
            prog_str = f"{self._step / self._steps * 100:.1f}" if self._steps else str(self._step)
            precision = [1, 1]
            
            table['done'] = [prog_str]
            table['time'] = [(time() - self._time_start) / 60]
            
            for mean_name in self._batch_metrics: table[mean_name] = []
            if self.verbose:
                for names in six.itervalues(self._batch_metrics):
                    for name in names:
                        table[name] = []
                        precision.append(self.precision)
                        
            for name, logs_ in six.iteritems(self._batch_logs):
                if name in table:
                    table[name].append(logs_[-1])
                    precision.append(self.precision)

            self._log(format_table(table, precision=precision, header=self._step == 1))


class TrainingStopper(Callback):
    def __init__(self, max_time=None, stop_file=None, verbose=1, logger=print):
        super(TrainingStopper, self).__init__()
        self.max_time = max_time
        self.stop_file = stop_file
        self.verbose = verbose
        self.logger = logger

    def on_train_begin(self, logs={}):
        self._time_start = time()

    def log(self, msg):
        if self.verbose: self.logger(msg)

    def on_epoch_end(self, batch, logs={}):
        if self.max_time is not None:
            elapsed = time() - self._time_start
            if elapsed > self.max_time:
                self.log('Stopping training after %.2fh' % (elapsed / 3600))
                self.model.stop_training = True

        if self.stop_file and os.path.isfile(self.stop_file):
            self.log('Stopping training due to stop file!')
            self.model.stop_training = True