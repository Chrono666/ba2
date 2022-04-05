import os
import matplotlib.pyplot as plt


def plot_training_graph():
    pass


def save_fig(fig_name, root_dir, sub_dir, fig_extension="png", resolution=300, tight_layout=True):
    """Save figure to filesystem.

    Args:
        fig_name (str): Name of figure.
        root_dir (str): Root directory.
        sub_dir (str): Subdirectory or directories.
        fig_extension (str): File extension.
        resolution (int): Resolution of figure.
        tight_layout (bool): Whether to use tight layout.
    """
    path = os.path.join(root_dir, sub_dir, fig_name + "." + fig_extension)
    os.makedirs(path, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
