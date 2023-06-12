from datasets import load_dataset
from transformers import AutoTokenizer

dataset_name = "jacobthebanana/sst5-mapped-extreme"
hf_model_name = "roberta-base"
label_map = ["negative", "neutral", "positive"]

def map_labels(example):
    example["label"] = label_map[example["label"]]
    return example

tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
ds = load_dataset(dataset_name)
print(ds["train"][0])
ds = ds.map(map_labels)
ds.push_to_hub("fkohankhaki/sst5-mapped-extreme-converted")
ds.save_to_disk("/h/fkohankh/fk-datasets/sst5-mapped-extreme-converted")
for split, dataset in ds.items():
    dataset.to_parquet(f"/h/fkohankh/fk-datasets/sst5-mapped-extreme-converted/sst5-mapped-extreme-converted-{split}.parquet")
print(ds["train"][0])