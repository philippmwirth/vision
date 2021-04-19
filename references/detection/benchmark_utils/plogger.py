from typing import *
from textwrap import wrap
from pathlib import Path
import os
from datetime import datetime

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import numpy as np

# copyright @malteebner

class NumpyFloatHandler(jsonpickle.handlers.BaseHandler):
    """
    Automatic conversion of numpy float  to python floats
    Required for jsonpickle to work correctly
    """

    def flatten(self, obj, data):
        """
        Converts and rounds a Numpy.float* to Python float
        """
        return round(obj, 6)


jsonpickle.handlers.registry.register(np.float, NumpyFloatHandler)
jsonpickle.handlers.registry.register(np.float32, NumpyFloatHandler)
jsonpickle.handlers.registry.register(np.float64, NumpyFloatHandler)


class ALEpisodeLog:
    def __init__(self, task_config: dict, sampler_config: dict):
        self.assert_input(task_config, sampler_config)
        self.task_config = task_config
        self.sampler_config = sampler_config
        self.metrics = {}
        self.start_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    def assert_input(self, task_config: dict, sampler_config: dict):
        assert isinstance(task_config, dict)
        assert isinstance(sampler_config, dict)
        assert 'name' in sampler_config.keys()

    def save_metrics(self, metrics: dict):
        assert 'no_labelled_samples' in metrics.keys()
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)


class ALBenchmarkPlogger:
    def __init__(self, filename="clothing_dataset_small_knn.json"):
        self.filename = filename

    def read_al_episode_logs_from_file(self) -> List[ALEpisodeLog]:
        # Read JSON data into the datastore variable
        with open(self.filename, 'r') as f:
            data_string = f.read()
            assert len(data_string) > 0, "The filename with the logs does not exist"
            datastore = jsonpickle.decode(data_string)
        return datastore

    def append_al_episode_logs_to_file(self, al_episode_log_list: List[ALEpisodeLog]):

        for al_episode_log in al_episode_log_list:
            al_episode_log.end_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        try:
            # Read JSON data into the datastore variable
            datastore = self.read_al_episode_logs_from_file()
            datastore += al_episode_log_list
        except Exception:
            datastore = al_episode_log_list

        # Writing JSON data
        dirname = os.path.dirname(os.path.abspath(self.filename))
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(self.filename, 'w') as f:
            data_string = jsonpickle.encode(datastore)
            f.write(data_string)
        debug_point = 0

    def delete_some_al_episode_logs(self, filter_function: Callable[[ALEpisodeLog, int], bool]):
        '''
        @param filter_function: if filterFunction(ALRunLog, index) returns True, ALRunLog is deleted
        @return: None
        '''

        al_episode_logs = self.read_al_episode_logs_from_file()
        al_episode_logs = [al_episode_log for index, al_episode_log in enumerate(al_episode_logs) if
                           not filter_function(al_episode_log, index)]

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            data_string = jsonpickle.encode(al_episode_logs, f)
            f.write(data_string)

    def plot_all_content_with_confidence_intervals(self, metric_name='test_acc', with_title: bool = True,
                                                   agent_names: List[str] = [],
                                                   sample_range: Tuple[int, int] = (0, np.inf),
                                                   plot_really: bool = True, filename_for_plot=None,
                                                   save_figure: bool = False):
        # define plots and legends
        run_representations = []
        al_episode_logs = self.read_al_episode_logs_from_file()
        for al_episode_log in al_episode_logs:
            run_representations += [(str(al_episode_log.sampler_config['name']),
                                     al_episode_log.metrics["no_labelled_samples"],
                                     al_episode_log.metrics[metric_name])]

        full_agent_names = list(set(representation[0] for representation in run_representations))
        if len(agent_names) == 0:
            agent_names = full_agent_names
        else:
            agent_names = list(set(agent_names) & set(full_agent_names))
        agent_names.sort(key=lambda name: name)

        fig = plt.figure(figsize=(6, 4), dpi=320)
        legends = []

        def mean_confidence_std(data_matrix, confidence: float = 0.95) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            '''
            @param data_matrix: shape: (noIterations, no_repetitions)
            @param confidence:
            @return: shapes: 5 times (noIterations,)
            '''
            means = np.mean(data_matrix, axis=1)
            stds = np.std(data_matrix, axis=1)
            no_repetitions = data_matrix.shape[1]
            deviation = stds * stats.t.ppf((1 + confidence) / 2., no_repetitions - 1) / (no_repetitions ** 0.5)
            return means, means - deviation, means + deviation, means - stds, means + stds

        color_cycle = plt.get_cmap("tab10")
        for i, agent_name in enumerate(agent_names):
            accuracies_list_dict = dict()
            for run_repr in [x for x in run_representations if x[0] == agent_name]:
                already_entered_no_labelled_samples = []
                for accuracy, no_labelled_samples in zip(run_repr[2], run_repr[1]):
                    if no_labelled_samples in already_entered_no_labelled_samples \
                            or no_labelled_samples < sample_range[0] \
                            or no_labelled_samples > sample_range[1]:
                        break
                    else:
                        if no_labelled_samples not in accuracies_list_dict.keys():
                            accuracies_list_dict[no_labelled_samples] = list()
                        accuracies_list_dict[no_labelled_samples].append(accuracy)
            bounds_list = list()
            for no_labelled_samples, accuracies_list in sorted(accuracies_list_dict.items()):
                accuracy_tensor = np.array(accuracies_list)[np.newaxis, :]
                bounds = mean_confidence_std(accuracy_tensor)
                bounds_list.append(bounds)
            bounds_tuple_array = [np.vstack(x)[:, 0] for x in zip(*bounds_list)]
            means, lower_bound, upper_bound, lower_bound_std, upper_bound_std = bounds_tuple_array

            no_labelled_samples_list = sorted(accuracies_list_dict.keys())
            with_confidence = any(lower_bound > lower_bound_std)
            if with_confidence:
                plt.fill_between(no_labelled_samples_list, lower_bound, upper_bound, color=color_cycle(i), alpha=.5)
            plt.fill_between(no_labelled_samples_list, lower_bound_std, upper_bound_std, color=color_cycle(i), alpha=.1)
            plt.plot(no_labelled_samples_list, means, color=color_cycle(i))
            legends += [agent_name]

        '''
        start plotting
        '''
        plt.legend(legends)

        title = "Task: " + os.path.basename(self.filename).replace(".json", "")
        # title += "\nEnv: " + str(al_episode_logs[-1].al_Parameters)
        title = "\n".join(wrap(title, 60))
        plt.xlabel('number of samples')
        plt.ylabel(metric_name)
        if "fashion" in title and "MNIST" not in title:
            title = title.replace("fashion", "fashion-MNIST")
        if with_title:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.grid()

        if save_figure:
            if filename_for_plot is None:
                filename_for_plot = self.filename
                filename_for_plot = filename_for_plot.replace(".json", ".png")
                filename_for_plot = filename_for_plot.replace("\ ", " ")
                filename_for_plot = filename_for_plot.replace(":", "_")
            plt.savefig(filename_for_plot, dpi=320)

        if plot_really:
            plt.show()
