import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from clu import metric_writers

FLAGS = flags.FLAGS


def create_logger(log_dir, enable):
    return metric_writers.create_default_writer(log_dir, just_logging=not enable)


def metrics_assign_group(metrics_dict, group, index=0):
    result = {}
    for k, v in metrics_dict.items():
        groups = k.split("/")
        abs_index = index % len(groups)
        groups = groups[:abs_index] + [group] + groups[abs_index:]
        new_k = "/".join(groups)
        result[new_k] = v
    return result


class LogAggregator:
    def __init__(
        self,
        keep_last=-1,
        provide_mean=True,
        provide_latest=True,
        reset_on_get=True,
    ):
        assert keep_last == -1 or keep_last > 0
        self.keep_last = keep_last
        self.provide_mean = provide_mean
        self.provide_latest = provide_latest
        self.reset_on_get = reset_on_get
        self.logs = []

        assert keep_last != -1 or reset_on_get  # otherwise memory will increase

    def add(self, new_logs):
        self.add_list([new_logs])

    def add_list(self, new_logs):
        new_logs = jax.device_get(new_logs)
        self.logs += new_logs
        if self.keep_last != -1 and len(self.logs) > self.keep_last:
            self.logs = self.logs[-self.keep_last :]

    def get_logs(self):
        metrics = {}

        if len(self.logs) != 0:
            if self.provide_mean:
                mean = jax.tree_map(lambda *args: np.mean(np.stack(args)), *self.logs)
                mean = dict(**mean)
                mean = metrics_assign_group(mean, "aggregated/mean", -1)
                metrics.update(mean)
                std = jax.tree_map(lambda *args: np.std(np.stack(args)), *self.logs)
                std = dict(**std)
                std = metrics_assign_group(std, "aggregated/std", -1)
                metrics.update(std)

                sample_size = jax.tree_map(lambda *args: len(args), *self.logs)
                sample_size = dict(**sample_size)
                sample_size = metrics_assign_group(sample_size, "aggregated/sample_size", -1)
                metrics.update(sample_size)

            if self.provide_latest:
                last = dict(**self.logs[-1])
                last = metrics_assign_group(last, "last", -1)
                metrics.update(last)

            if self.reset_on_get:
                self.logs = []

        return metrics
