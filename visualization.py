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
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import colors as mpl_colors
import matplotlib.cm as cm


"""
Matplotlib plots.
"""


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


def plot_bboxes_on_image(image, *bbox_instances, bbox_format="xy1xy2", labels=None):
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
    fig.set_size_inches(6, 6)

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


"""
OpenCV2 drawings.
"""


def draw_circles_on_image(image, points, point_colors=None, default_color=(255, 0, 0)):
    """
    Draw points on a given image with different options of coloring.

    :image:
    :points: List of point in [(x, y), ...] format.
    :point_colors: List of color for each point.
    :default_color: RGB tuple of color if all points have same color.
    """

    if len(image.shape) == 2:
        img = np.stack((image,)*3, axis=-1)
    else:
        img = image.copy()

    if not point_colors:
        point_colors = [default_color for _ in  range(points.shape[0])]

    for p, c in zip(points, point_colors):
        x, y = p
        radius = 5
        thickness=-1
        color = c[:3]

        img = cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        img = cv2.circle(img, (int(x), int(y)), radius, (0, 0, 0), 0)

    return img


# TODO add coloring options like in draw_circle_on_image function
def draw_bboxes_on_image(image, *bbox_instances, bbox_format="xy1xy2"):
    """
    Draw bounding boxes on image.

    :image:
    :bbox_instances: Bboxes with format specified in bbox_format.
    :bbox_format: Format of how bbox is saved. E.g. xy1xy2 = (xmin, ymin, xmax, ymax)
    """

    colors = plt.get_cmap("Set1").colors
    colors = tuple(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))

    assert len(bbox_instances) < len(colors), f"Only {len(colors)} bbox instances supported."

    gray_image = cv2.cvtColor(image, np.array(1), cv2.COLOR_GRAY2BGR)
    thickness = 2

    for i, bboxes in enumerate(bbox_instances):
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = parse_bbox(bbox, bbox_format, "xy1xy2")
            gray_image = cv2.rectangle(gray_image,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)), colors[i], thickness)

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

# Must be at the end to workaround cyclic import with statistics.py
import statistics
