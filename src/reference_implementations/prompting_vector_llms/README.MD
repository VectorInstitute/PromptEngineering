# Prompting Vector's Large Language Models

This folder contains a number of notebooks and reference implementations working with Vector's toolkit for interacting with LLMs hosted on Vector's cluster. The current models hosted on our cluster are:

__NOTE__: Due to their size, and depending on the size sequence input, the largest models may take some time to respond. In addition, the model can be sensitive to Out-of-memory errors if a large batch of inputs and activations are requested. So please be sure to batch your inputs, as is done in the notebooks.

1) OPT-175B: Currently the largest model in the OPT family at 175B parameters.
2) OPT-6.7B: One of the moderate generative models in the OPT family. It does not have as significant generalization capabilties but is much faster to respond and can handle larger requests.
3) Galactica-120B: This model shares the same architecture as OPT but a different underlying training corpus. During training many academic and scientific texts were also included.



## LLM Prompting Examples

This folder gathers a number of example demonstrations of LLM prompting. There are four notebooks considering different tasks. The notebooks explore the effects of manual prompt optimization, varying instruction prompts, zero- and few-shot performance. The tasks considered are:

1) Summarization
2) Classification
3) BoolQ (booling question answering)
4) Basic Translation (French -> English)

The notebooks are loosely ordered, in the sense that they build somewhat on each other. The rough ordering follows the enumeration above. That is, `Summarization`, `Classification`, `BoolQ`, and `Translation`

### Summarization

This notebook explores the summarization capabilties of OPT on a small sample of news articles. This includes the affects of variations on instructions, how postprocessing might be used to prepare the summaries, and the various behaviors depending on how the prompt is structured.

### Classification

The notebook considers the task of categorizing news into one of four categories. The task is based on the AG News dataset, which has news stories belonging to the categories of World, Sports, Business, and Science/Technology. The notebook measures performance on a small sample from the AG news test set. Manual prompt optimization, zero-shot, and few-shot prompts are considered. The notebook also explores the concept of "verbalizers." That is, how do we interpret answers given by the notebook. Specifically, we consider pure response and logits extraction associated with a chosen label space.

### BoolQ

This notebook implements an example of performing question answering based on a provided context. The BoolQ task consists of a subject, passage, and question based on the passage with a boolean answer. The question always has a correct response based on the context. We consider zero-shot and few-shot performance on a small sample of the full BoolQ dataset. The notebook also compares performance using two different label spaces.


### Translation

The notebook is constructed to provide a proof of concept in translation through prompting of OPT. OPT is __not__ trained on a multilingual dataset, but has been shown to be able to perform some translations. This notebook demonstrates these capabilities in zero- and few-shot settings. However, the model isn't particularly strong, as measured by the bleu scores on a small sample from the NMT14 fr-en test set.
