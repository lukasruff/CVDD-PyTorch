import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joypy


def plot_joyplot(data, title, export_pdf=False, show=False):
    """
    :param data: np.array with scores [n_samples, n_attention_heads]
    :param title: title for plot
    :param export_pdf: export file
    :param show: show plot
    """

    large, med, small = 22, 16, 12
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    # convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = joypy.joyplot(df, figsize=(14, 10))

    # Decoration
    plt.title(title, fontsize=22)

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    if show:
        plt.show()

    return
