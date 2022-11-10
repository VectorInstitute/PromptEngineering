"""
This module implements the required seqio task for reading a text file containing
sentences for the binary sentiment analysis task.
"""

import pandas as pd
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import Dataset
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
from t5.data.glue_utils import get_glue_metric, get_glue_postprocess_fn, get_glue_text_preprocessor


def load_input_dataset(filepath: str) -> tf.data.Dataset:
    """Easy way to read local csv into the tensorflow dataset."""
    df = pd.read_csv(filepath, delimiter=",", index_col="idx")
    ds = Dataset.from_pandas(df)
    ds.set_format(type="tensorflow", columns=["idx", "sentence", "label"])
    dataset = {x: ds[x] for x in ["idx", "sentence", "label"]}
    tfdataset = tf.data.Dataset.from_tensor_slices(dataset)
    return tfdataset


for _, feats in features.MODEL_TO_FEATURES.items():
    for b in tfds.text.glue.Glue.builder_configs.values():
        # only select the binary sentiment analysis task (sst2).
        if b.name == "sst2":
            postprocess_fn = get_glue_postprocess_fn(b)
            metric_fns = get_glue_metric(b.name)
            seqio.TaskRegistry.add(
                "example_binary_sentiment_analysis",
                source=load_input_dataset("example_input_sentences.csv"),
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
