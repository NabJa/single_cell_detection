"""
Plotting library for object detection and model evaluation tasks.
"""

from os.path import join, basename
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import cv2

import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.pylab import setp
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib import colors as mpl_colors

import statistics
"""
Matplotlib plots.
"""


def heatmap(data, title="", xlabel="", ylabel="", xticks=None, yticks=None):
    """
    Plots a heatmap of the given data.

    :param data: Numpy array of shape (row, col)
    """

    xticks = np.round(np.linspace(0, 0.9, data.shape[0]), 2)
    yticks = np.round(np.linspace(0, 0.9, data.shape[1]), 2)

    fig, ax = plt.subplots()
    ax.imshow(data)

    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_xticklabels(xticks, fontsize=15)
    ax.set_yticklabels(yticks, fontsize=15)

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f"{data[i, j]:.3f}",
                           ha="center", va="center", color="black", fontsize=15)

    ax.set_title(title, fontsize=25)
    fig.set_size_inches(12, 12)
    fig.tight_layout()
    plt.show()


def plot_image_distribution(image, title):
    """
    Plot 3D image distribution with pixel values in z-axis.
    """

    height, width, *_ = image.shape
    x, y = np.linspace(0, 1, height), np.linspace(0, 1, width)

    fig = go.Figure(data=[go.Surface(z=image, x=x, y=y)])
    fig.update_layout(title=title, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


def density_scatter(x, y, **kwargs):
    """
    Plots a density scatter (2D distribution).
    """
    x, y = np.array(x), np.array(y)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.scatter(x, y, c=z, edgecolor='', **kwargs)
    plt.show()


def plot_bboxes_on_image(image, *bbox_instances, bbox_format="xy1xy2", labels=None, title=""):
    """
    Plot bounding boxes on image.

    :image:
    :bbox_instances: Bboxes with format specified in bbox_format.
    :bbox_format: Format of how bbox is saved. E.g. xy1xy2 = (xmin, ymin, xmax, ymax)
    :labels: Legend labels for given bboxes.
    """

    colors = plt.get_cmap("Set1").colors

    assert len(bbox_instances) < len(colors), f"Only {len(colors)} bbox instances supported."

    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 16)
    ax.set_title(title)

    # Display the image
    ax.imshow(image, cmap="gray")

    legend_lines = []
    labels = labels or [str(i) for i in range(len(bbox_instances))]

    for i, bboxes in enumerate(bbox_instances):
        legend_lines.append(Line2D([0], [0], color=colors[i], lw=4))
        for bbox in bboxes:
            x, y, w, h = parse_bbox(bbox, bbox_format, "xywh")
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    ax.legend(legend_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_simple_boxplot(boxes, colors=None, labels=None, title="", y_label="", y_range=None, save=None):
    """
    Simple Boxplot for every box in boxes.

    :param boxes: List of lines to draw. Format: [[values1], [values2], ...]
    :param colors: List of colors. Format: [color1, color2, ...]
    :param labels: List of labels. Format: [label1, label2, ...]
    :param save: Path to file the figure should be saved to. Default: Only show plot.
    """
    colors = list(mpl_colors.TABLEAU_COLORS.values())[:len(boxes)] if not colors else colors
    labels = list(range(len(boxes))) if not labels else labels

    plt.figure(figsize=(16, 8))
    plt.title(title, fontsize=25)
    for i, (b, c, l) in enumerate(zip(boxes, colors, labels)):
        box = plt.boxplot(b, positions=[i], widths=0.5)
        for box_property in box.values():
            setp(box_property, color=c, lw=4)
    plt.xticks(ticks=list(range(len(boxes))), labels=labels, fontsize=20)
    plt.yticks(y_range, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.grid(axis="y")
    plt.savefig(save) if save else plt.show()


def plot_simple_lines(lines, colors=None, labels=None, title="", x_label="", y_label="", save=None):
    """
    Simple multiple lines plot.

    :param lines: List of lines to draw. Format: [[values1], [values2], ...]
    :param colors: List of colors. Format: [color1, color2, ...]
    :param labels: List of labels. Format: [label1, label2, ...]
    :param save: Path to file the figure should be saved to. Default: Only show plot.
    """
    colors = list(mpl_colors.TABLEAU_COLORS.values())[:len(lines)] if not colors else colors
    labels = list(range(len(lines))) if not labels else labels

    plt.figure(figsize=(16, 8))
    plt.title(title, fontsize=25)
    for d, c, l in zip(lines, colors, labels):
        plt.plot(d, linewidth=5, color=c, label=l, alpha=0.8)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(save) if save else plt.show()


"""
OpenCV2 drawings.
"""


def draw_circles_on_image(image, *point_instances, colors=None):
    """
    Draw points on a given image with different options of coloring.

    :image:
    :points: Instances of points in [(x, y), ...] format.
    """

    if np.array(colors).any():
        new_colors = values_to_rgb(colors)
    else:
        new_colors = plt.get_cmap("Set1").colors
    new_colors = tuple(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), new_colors))

    gray_image = cv2.cvtColor(image, np.array(1), cv2.COLOR_GRAY2BGR)
    radius = 5
    thickness = -1

    for i, points in enumerate(point_instances):
        for j, p in enumerate(points):
            x, y = p
            color = new_colors[j] if np.array(colors).any() else new_colors[i]
            gray_image = cv2.circle(gray_image, (int(x), int(y)), radius, color, thickness)
            gray_image = cv2.circle(gray_image, (int(x), int(y)), radius, (0, 0, 0), 0)

    return gray_image


def draw_bboxes_on_image(image, *bbox_instances, colors=None, bbox_format="xy1xy2"):
    """
    Draw bounding boxes on image.

    :image:
    :bbox_instances: Bboxes with format specified in bbox_format.
    :param colors: Numpy array of colors.
    :bbox_format: Format of how bbox is saved. E.g. xy1xy2 = (xmin, ymin, xmax, ymax)
    """

    if np.array(colors).any():
        new_colors = values_to_rgb(colors)
    else:
        new_colors = plt.get_cmap("Set1").colors
    new_colors = tuple(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), new_colors))

    gray_image = cv2.cvtColor(image, np.array(1), cv2.COLOR_GRAY2BGR)
    thickness = 2

    for i, bboxes in enumerate(bbox_instances):
        for j, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = parse_bbox(bbox, bbox_format, "xy1xy2")
            color = new_colors[j] if np.array(colors).any() else new_colors[i]
            gray_image = cv2.rectangle(gray_image,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)), color, thickness)

    return gray_image


# TODO command line support
# TODO generalize write video
def write_video_from_csv(csv_path, image_path, out_path, bbox_key="bbox20", fps=10):
    """
    Write a video with bboxes from csv file with columns: [Frame, X_Position, Y_Position]
    """
    df = pd.read_csv(csv_path, engine='python')

    images = glob(join(image_path, "*.png"))

    height, width, *_ = cv2.imread(images[0]).shape
    video = cv2.VideoWriter(join(out_path, f'{basename(image_path)}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in np.unique(df.Frame):
        image = cv2.imread(images[frame])
        bboxes = list(df[df.Frame == frame][bbox_key])
        image = draw_bboxes_on_image(image, bboxes)
        video.write(image)
    video.release()


"""
PLOTLY GRAPHS
"""


def plotly_image_slider(images, ticks, slider_prefix="Distance < "):
    """
    Plots all images with a slider named after ticks.
    """

    fig = go.Figure()

    for img in images:
        fig.add_trace(
            go.Image(z=img, visible=False)
            )

    fig.data[0].visible = True

    steps = []
    for i, t in enumerate(ticks):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=t
        )
        step["args"][1][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": slider_prefix},
        pad={"t": 50},
        steps=steps
    )]


    fig.update_layout(
        template="none",
        sliders=sliders,
    )

    return fig


def plot_precision_recall_curves(gt, *args, title="Precision-Recall curve", names=None):
    """
    Plotly plot of precision recall curve with slider for different IoU thresholds.

    gt: Ground truth bounding boxes [N, (xmin, ymin, xmax, ymax)]
    args: Prediction bounding boxes [M, (xmin, ymin, xmax, ymax)]
    """

    if not names or len(names) != len(args):
        names = list(range(len(args)))

    colors = px.colors.qualitative.D3

    fig = go.Figure()

    ious = np.arange(0.1, 1, 0.1)

    # Add traces, one for each slider step
    for arg, pred in enumerate(args):
        for iou in ious:
            mAP, precisions, recalls, _ = statistics.compute_ap(pred, gt, iou)
            mAP = str(np.round(mAP, 3)).ljust(5, '0')

            fig.add_trace(
                go.Scatter(
                    visible=bool(iou == 0.5),
                    line=dict(color=colors[arg], width=4),
                    name=f"{names[arg]} mAP={mAP}",
                    x=recalls,
                    y=precisions))

    steps = []
    for i, iou in enumerate(ious):

        slider_args = [False for x in range(len(fig.data))]

        # Given IoU set to True
        for j in range(len(args)):
            slider_args[i+j*len(ious)] = True

        step = dict(
            method="restyle",
            args=["visible", slider_args],
            label=np.round(iou, 2),
        )
        steps.append(step)

    sliders = [dict(
        active=4,
        currentvalue={"prefix": "Current IoU: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title=title,
        template="none",
        sliders=sliders,
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
        xaxis_title='Recall',
        yaxis_title='Precision'
    )

    return fig


def plotly_precision_recall_slider(precisions, recalls, ticks, slider_prefix="Distance < ", title="Precision-Recall curve", names=None):
    """
    :precisions: List of precision values. E.g. [[exp1], [exp2]]
    :recalls: List of recall values. E.g. [[exp1], [exp2]]
    :ticks: List of tick names. E.g. [1, 2]

    Plots Precision-Recall curves for different ticks.
    """

    if not names or len(names) != len(precisions):
        names = list(range(len(precisions)))

    fig = go.Figure()

    for prec, rec, t in zip(precisions, recalls, ticks):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="blue", width=4),
                name="",
                x=rec,
                y=prec))

    fig.data[0].visible = True

    steps = []
    for i, t in enumerate(ticks):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=t
        )
        step["args"][1][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": slider_prefix},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title=title,
        template="none",
        sliders=sliders,
    )
    return fig


"""
Helper functions.
"""


def parse_bbox(bbox, bbox_format, res_format="xywh"):
    """
    Restructure bbox to given format.
    """

    if bbox_format == "xywh":
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
    elif bbox_format == "xy1xy2":
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
    elif bbox_format == "yx1yx2":
        ymin, xmin, ymax, xmax = bbox
        width = xmax - xmin
        height = ymax - ymin
    else:
        raise NotImplementedError(f"{bbox_format} not supported.")

    if res_format == "xywh":
        return xmin, ymin, width, height
    if res_format == "xy1xy2":
        return xmin, ymin, xmax, ymax
    if res_format == "yx1yx2":
        return ymin, xmin, ymax, xmax
    else:
        raise NotImplementedError(f"{res_format} not supported.")


def values_to_rgb(values):
    minimum = np.min(values)
    maximum = np.max(values)

    norm = mpl_colors.Normalize(vmin=minimum, vmax=maximum, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    val_colors = np.array(list(map(mapper.to_rgba, values))) * 255
    val_colors = val_colors[:, :3]
    return val_colors.tolist()


if __name__ == "__main__":
    pass
