"""
Loading utils for tf_records
"""

import io
import numpy as np
import tensorflow as tf
from PIL import Image

tf.compat.v1.enable_eager_execution()


def load_tf_dataset(path):
    raw_image_dataset = tf.data.TFRecordDataset(str(path))

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    images = []
    bboxes = []
    names = []
    for image_features in parsed_image_dataset:
        # Read encoded image

        image_filename = image_features['image/filename'].numpy()

        image_raw = image_features['image/encoded'].numpy()
        image = np.array(Image.open(io.BytesIO(image_raw)).convert('L'))

        height = np.array(image_features['image/height'].numpy())
        width = np.array(image_features['image/width'].numpy())

        xmin = np.array(image_features['image/object/bbox/xmin'].numpy()) * width
        ymin = np.array(image_features['image/object/bbox/ymin'].numpy()) * height
        xmax = np.array(image_features['image/object/bbox/xmax'].numpy()) * width
        ymax = np.array(image_features['image/object/bbox/ymax'].numpy()) * height

        images.append(image)
        bboxes.append(np.stack((xmin, ymin, xmax, ymax), axis=-1))
        names.append(image_filename)
    return {"names": names, "images": images, "bboxes": bboxes}


def tf_dataset_generator(path):
    raw_image_dataset = tf.data.TFRecordDataset(str(path))

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:

        image_filename = image_features['image/filename'].numpy()

        image_raw = image_features['image/encoded'].numpy()
        image = np.array(Image.open(io.BytesIO(image_raw)).convert('L'))

        height = np.array(image_features['image/height'].numpy())
        width = np.array(image_features['image/width'].numpy())

        xmin = np.array(image_features['image/object/bbox/xmin'].numpy()) * width
        ymin = np.array(image_features['image/object/bbox/ymin'].numpy()) * height
        xmax = np.array(image_features['image/object/bbox/xmax'].numpy()) * width
        ymax = np.array(image_features['image/object/bbox/ymax'].numpy()) * height
        bboxes = np.stack((xmin, ymin, xmax, ymax), axis=-1)

        yield {"name": image_filename, "image": image, "bboxes": bboxes}


def _parse_image_function(example_proto):

    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing = True),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)
