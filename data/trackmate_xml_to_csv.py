"""
Script to transform TrackMate output XML to csv.
"""

import argparse
import os
from os.path import join
import xml.etree.ElementTree as ET
import pandas as pd


def extract_points_from_trackmate_xml(path, bbox_sizes):
    """
    Parses TrackMate ouput xml file.

    path: Path to XML file.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    x_positions = []
    y_positions = []
    frames = []

    # Key=bbox_size, value=list of bboxes
    bboxes = {}

    for spot in root.iter("Spot"):
        x = float(spot.attrib["POSITION_X"])
        y = float(spot.attrib["POSITION_Y"])
        frame = int(float(spot.attrib["POSITION_T"]))

        x_positions.append(x)
        y_positions.append(y)
        frames.append(frame)

        for bbox_size in bbox_sizes:
            bbox = apply_bbox_on_point(x, y, int(bbox_size))
            bbox_list = bboxes.setdefault(bbox_size, [])
            bbox_list.append(bbox)
            bboxes[bbox_size] = bbox_list

    points_df = pd.DataFrame({
        "Frame": frames,
        "X_Position": x_positions,
        "Y_Position": y_positions,
    })

    for bbox_size, bbox_list in bboxes.items():
        bbox_name = f"bbox{bbox_size}"
        points_df[bbox_name] = bbox_list

    return points_df


def apply_bbox_on_point(x, y, size=20):
    """
    Create a bbox of size around a point.

    size: size of bbox
    """
    offset = size/2
    bbox_x_start = int(x - offset) if int(x - offset) >= 0 else 0
    bbox_y_start = int(y - offset) if int(y - offset) >= 0 else 0

    bbox_x_end = int(x + offset)
    bbox_y_end = int(y + offset)

    return bbox_x_start, bbox_y_start, bbox_x_end, bbox_y_end


def _xml_path(path):
    if os.path.exists(path) and os.path.basename(path).endswith("xml"):
        return path
    raise NotADirectoryError(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create df from TrackMate XML.')
    parser.add_argument('--path', type=_xml_path, help="Path to TrackMate xml.")
    parser.add_argument('--bbox_sizes', nargs='*', help="Size of generated bboxes. Default=20.")
    parser.add_argument('--out', help="Output name")
    args = parser.parse_args()

    if not args.bbox_sizes:
        args_size = [20]
    else:
        args_size = args.bbox_sizes

    if not args.out:
        out_file_name = os.path.basename(args.path).split(".")[0]
        out_name = f"{join(os.path.dirname(args.path), out_file_name)}.csv"
    else:
        out_name = args.out

    df = extract_points_from_trackmate_xml(args.path, args_size)
    print(f"Writing csv to {out_name}")
    df.to_csv(out_name)
