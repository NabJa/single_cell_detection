"""
Functions to transform predictions into TrackMate XML.
"""
import argparse

import os
from os.path import join, isdir
from collections import OrderedDict
from glob import glob
import pickle

import numpy as np
from tqdm import tqdm
import xmltodict

import cv2
import predict_on_series as predictor

def prepare_template_xml(image_path, template_path="Template.xml"):

    # Using a minimal Template.xml located next to this scrpit as a basis
    # 'doc' will hold the XML-Tree as a dictionary
    with open(template_path) as template:
        doc = xmltodict.parse(template.read())

    # List images to be processed, filtered by ending
    images = glob(join(image_path, "*.png"))

    img = cv2.imread(images[0], 0)

    height, width = img.shape[:2]

    # Add dimensions and number of images to ImageData and BasicSettings in XML
    doc['TrackMate']['Settings']['ImageData']['@filename'] = images[0]
    doc['TrackMate']['Settings']['ImageData']['@folder'] = image_path
    doc['TrackMate']['Settings']['ImageData']['@width'] = str(width)
    doc['TrackMate']['Settings']['ImageData']['@height'] = str(height)
    doc['TrackMate']['Settings']['ImageData']['@nframes'] = str(len(images))

    # TODO sure about -1??
    doc['TrackMate']['Settings']['BasicSettings']['@xend'] = str(width)
    doc['TrackMate']['Settings']['BasicSettings']['@yend'] = str(height)
    doc['TrackMate']['Settings']['BasicSettings']['@tend'] = str(len(images))

    return doc


def points_to_xml(image_path, predout, xmlout, min_size=15, min_score=0.5):
    """
    Write a trackmate XML for given predictions.

    :image_path: path to image folder containing input images in png format.
    :predout: path to folder containing predictions in pickle format
    :xmlout: path to write the xml
    :min_size: point radius
    :min_score: min detection score for cutoff
    """

    doc = prepare_template_xml(image_path)

    for img_no, pred_path in enumerate(tqdm(glob(join(predout, "*.p")))):

        pred = pickle.load(open(pred_path, "rb"))
        detection_boxes = pred.get("detection_boxes")
        detection_score = pred.get("detection_scores")

        detection_boxes = detection_boxes[detection_score >= min_score]
        detection_score = detection_score[detection_score >= min_score]

        points = bboxes_to_points(detection_boxes)
        image_dir = pred.get("image_dir")

        for cell_no, (point, score) in enumerate(zip(points, detection_score)):

            pred_id = "{:05d}{:05d}".format(img_no, cell_no)
            name = "{:05d}-{:05d}".format(img_no, cell_no)
            add_cell(doc, cell_id=pred_id, name=image_dir, position_x=str(point[0]), position_y=str(point[1]), frame=img_no, radius=min_size, quality=score)

    with open(xmlout, 'w') as result_file:
        result_file.write(xmltodict.unparse(doc, pretty=True))


def bboxes_to_points(bboxes):
    """
    Takes bboxes in [ymin, xmin, ymax, xmax] format and transorms them to points in [x, y] format.

    :bboxes: [N, (ymin, xmin, ymax, xmax)] array of bboxes
    """
    ymins, xmins = bboxes[:, 0], bboxes[:, 1]
    ymaxs, xmaxs = bboxes[:, 2], bboxes[:, 3]

    widths = (xmaxs - xmins) / 2
    heights = (ymaxs - ymins) / 2

    x_coords = xmins+widths
    y_coords = ymins+heights

    points = np.stack((x_coords, y_coords), axis=-1)
    return points


def add_cell(xml, cell_id="", name="", position_x="", position_y="", frame=0, radius=0, quality=1):
    cell = cellToDict(cell_id, name, position_x, position_y, frame, radius, quality)

    n_spots = int(xml['TrackMate']['Model']['AllSpots']['@nspots'])

    if "SpotsInFrame" in xml['TrackMate']['Model']['AllSpots']:
        spots_in_frame = xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        if isinstance(spots_in_frame, list):
            found = False
            for spot in spots_in_frame:
                if int(spot['@frame']) == frame:
                    spot['Spot'].append(cell)
                    found = True
                    break
            if not found:
                spots_in_frame.append(OrderedDict({
                    '@frame': str(frame),
                    'Spot': [cell]
                }))
        elif isinstance(spots_in_frame, OrderedDict):
            if spots_in_frame['@frame'] == str(frame):
                spots_in_frame['Spot'].append(cell)
            else:
                xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = [spots_in_frame]
                xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'].append(
                    OrderedDict({
                        '@frame': str(frame),
                        'Spot': [cell]
                    })
                )
    else:
        xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = OrderedDict({
            "@frame": str(frame),
            "Spot": [cell]
        })

    xml['TrackMate']['Model']['AllSpots']['@nspots'] = str(n_spots + 1)


def cellToDict(ID, NAME, POSITION_X, POSITION_Y, FRAME, RADIUS, QUALITY):
    return OrderedDict({
        '@ID': ID,                          # Must be numeric
        '@name': NAME,                      # Can be any string
        '@QUALITY': str(QUALITY),
        '@POSITION_X': str(POSITION_X),
        '@POSITION_Y': str(POSITION_Y),
        '@POSITION_Z': "0.0",
        '@POSITION_T': str(FRAME),
        '@FRAME': str(FRAME),
        '@RADIUS': str(RADIUS),
        '@VISIBILITY': "1",
        '@MANUAL_COLOR': "-10921639",
        '@MEAN_INTENSITY': "255.00",
        '@MEDIAN_INTENSITY': "255.00",
        '@MIN_INTENSITY': "255.00",
        '@MAX_INTENSITY': "255.00",
        '@TOTAL_INTENSITY': "255.00",
        '@STANDARD_DEVIATION': "0.0",
        '@ESTIMATED_DIAMETER': str(RADIUS * 2),
        '@CONTRAST': "0.0",
        '@SNR': "1.0"
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", "-i", required=True, type=str)
    parser.add_argument("--model_dir", "-m", required=True, type=str)
    parser.add_argument("--output_dir", "-o", required=True, type=str)

    args = parser.parse_args()

    if not isdir(args.image_dir):
        raise FileNotFoundError(f"No such directory: {args.image_dir}")
    if not isdir(args.model_dir):
        raise FileNotFoundError(f"No such directory: {args.model_dir}")
    if not isdir(args.output_dir):
        print(f"WARNING: Generating output directory: {args.output_dir}")
        os.mkdir(args.output_dir)

    # If predictions do not exist, predict.
    if len(glob(join(args.output_dir, "*.p"))) != len(glob(join(args.image_dir, "*.png"))):
        predictor.main(args.image_dir, args.model_dir, args.output_dir, "pickle")

    xml_out_name = join(args.image_dir, "trackmate.xml")
    points_to_xml(args.image_dir, args.output_dir, xml_out_name)
