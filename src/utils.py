import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple

def save_figure_to_png(fig, outpath):
    fig.savefig(outpath, bbox_inches='tight')
    return outpath

def make_basic_timeseries_plot(signal, sr=1.0, title='Signal') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8,3))
    t = np.arange(len(signal))/sr
    ax.plot(t, signal)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig
