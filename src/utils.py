import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

log = logging.getLogger(__name__)


class timeit_context:
    """
    Usage:

    ```python
    with timeit_context() as timer:
        ... # code block to be measured
    ```
    """

    def _log(self, msg):
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        Sets the start time and logs an optional message indicating the start of the code block execution.

        Args:
            msg: str, optional message to log
        """
        self.start_time = time.time()
        if self.msg is not None:
            self._log(self.msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calculates the elapsed time and logs it, along with an optional message indicating the end of the code block execution.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._log(f"Elapsed time: {elapsed_time:.2f}s")


# ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model: nn.Module):
    """
    Prints the number of non-zero parameters, total parameters, percentage of non-zero parameters, and compression rate of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to print the non-zero parameters of.

    Returns:
        float: The percentage of non-zero parameters in the model.
    """
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(
            f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
        )
    print(
        f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
    )
    return nonzero / total


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


# ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    """
    Creates a directory if it does not already exist.

    Args:
        directory (str): The path to the directory to create.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# FIXME
def plot_train_test_stats(
    stats,
    epoch_num,
    key1="train",
    key2="test",
    key1_label=None,
    key2_label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    yscale=None,
    ylim_bottom=None,
    ylim_top=None,
    savefig=None,
    sns_style="darkgrid",
):
    assert len(stats[key1]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key1, len(stats[key1]), epoch_num)
    assert len(stats[key2]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key2, len(stats[key2]), epoch_num)

    plt.clf()
    sns.set_style(sns_style)
    x_ticks = np.arange(epoch_num)

    plt.plot(x_ticks, stats[key1], label=key1_label)
    plt.plot(x_ticks, stats[key2], label=key2_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fancybox=True)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()
