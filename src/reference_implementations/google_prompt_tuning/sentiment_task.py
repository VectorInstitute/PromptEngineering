"""
This module implements the required seqio task for reading a text file containing
sentences for the binary sentiment analysis task.
"""
import os
from typing import Dict

import pandas as pd
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import Dataset
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
from t5.data.glue_utils import get_glue_metric, get_glue_postprocess_fn, get_glue_text_preprocessor


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value: tf.Tensor) -> tf.train.BytesList:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: tf.Tensor) -> tf.train.FloatList:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: tf.Tensor) -> tf.train.Int64List:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(f0: tf.Tensor, f1: tf.Tensor, f2: tf.Tensor) -> bytes:
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "idx": _int64_feature(f0),
        "sentence": _bytes_feature(f1),
        "label": _int64_feature(f2),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(row: Dict[str, tf.Tensor]) -> tf.string:
    """Define the serialize function for tensorflow dataset mapper."""
    tf_string = tf.py_function(
        serialize_example,
        (row["idx"], row["sentence"], row["label"]),  # Pass these args to the above function.
        tf.string,
    )  # The return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar.


def load_input_dataset(filedir: str, filename: str) -> str:
    df = pd.read_csv(os.path.join(filedir, filename), delimiter=",", index_col="idx")
    ds = Dataset.from_pandas(df)
    ds.set_format(type="tensorflow", columns=["idx", "sentence", "label"])
    dataset = {x: ds[x] for x in ["idx", "sentence", "label"]}
    tfdataset = tf.data.Dataset.from_tensor_slices(dataset)
    serialized_dataset = tfdataset.map(tf_serialize_example)
    tfrecord_name = os.path.join(filedir, filename.rstrip(".csv") + ".tfrecord")
    writer = tf.data.experimental.TFRecordWriter(tfrecord_name)
    writer.write(serialized_dataset)
    return tfrecord_name


filedir = "./src/reference_implementations/google_prompt_tuning/resources"
filename = "example_input_sentences.csv"
tfrecord_name = load_input_dataset(filedir, filename)

for _, feats in features.MODEL_TO_FEATURES.items():
    for b in tfds.text.glue.Glue.builder_configs.values():
        # only select the binary sentiment analysis task (sst2).
        if b.name == "sst2":
            postprocess_fn = get_glue_postprocess_fn(b)
            metric_fns = get_glue_metric(b.name)
            seqio.TaskRegistry.add(
                "example_binary_sentiment_analysis",
                source=seqio.TFExampleDataSource(
                    split_to_filepattern={"test": tfrecord_name},
                    feature_description={
                        "idx": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                        "sentence": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                        "label": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                    },
                ),
                preprocessors=[
                    get_glue_text_preprocessor(b),
                    pt_preprocessors.remove_first_text_token,
                    seqio.preprocessors.tokenize,
                    seqio.preprocessors.append_eos_after_trim,
                ],
                postprocess_fn=postprocess_fn,
                metric_fns=metric_fns,
                output_features=feats,
            )
