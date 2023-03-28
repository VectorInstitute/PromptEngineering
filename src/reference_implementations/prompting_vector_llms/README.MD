# Prompting Vector's Large Language Models

This folder contains a number of notebooks and reference implementations working with Vector's toolkit for interacting with LLMs hosted on Vector's cluster. The current models hosted on our cluster are:

__NOTE__: Due to their size, and depending on the size sequence input, the largest models may take some time to respond. In addition, the model can be sensitive to Out-of-memory errors if a large batch of inputs and activations are requested. So please be sure to batch your inputs, as is done in the notebooks.

1) OPT-175B: Currently the largest model in the OPT family at 175B parameters.
2) OPT-6.7B: One of the moderate generative models in the OPT family. It does not have as significant generalization capabilties but is much faster to respond and can handle larger requests.
3) Galactica-120B: This model shares the same architecture as OPT but a different underlying training corpus. During training many academic and scientific texts were also included.



## LLM Prompting Examples

This folder gathers a number of example demonstrations of LLM prompting. There are five notebooks considering different tasks. The notebooks explore the effects of manual prompt optimization, varying instruction prompts, zero- and few-shot performance. The tasks considered are:

1) Summarization
2) Classification
3) BoolQ (booling question answering)
4) Basic Translation (French -> English)
5) Aspect-Based Sentiment Analysis

The notebooks are loosely ordered, in the sense that they build somewhat on each other. The rough ordering follows the enumeration above. That is, `Summarization`, `Classification`, `BoolQ`, `Translation`, and `ABSA`

### Summarization

This notebook explores the summarization capabilties of OPT on a small sample of news articles. This includes the affects of variations on instructions, how postprocessing might be used to prepare the summaries, and the various behaviors depending on how the prompt is structured.

### Classification

The notebook considers the task of categorizing news into one of four categories. The task is based on the AG News dataset, which has news stories belonging to the categories of World, Sports, Business, and Science/Technology. The notebook measures performance on a small sample from the AG news test set. Manual prompt optimization, zero-shot, and few-shot prompts are considered. The notebook also explores the concept of "verbalizers." That is, how do we interpret answers given by the notebook. Specifically, we consider pure response and logits extraction associated with a chosen label space.

### BoolQ

This notebook implements an example of performing question answering based on a provided context. The BoolQ task consists of a subject, passage, and question based on the passage with a boolean answer. The question always has a correct response based on the context. We consider zero-shot and few-shot performance on a small sample of the full BoolQ dataset. The notebook also compares performance using two different label spaces.


### Translation

The notebook is constructed to provide a proof of concept in translation through prompting of OPT. OPT is __not__ trained on a multilingual dataset, but has been shown to be able to perform some translations. This notebook demonstrates these capabilities in zero- and few-shot settings. However, the model isn't particularly strong, as measured by the bleu scores on a small sample from the NMT14 fr-en test set.

### Aspect-Based Sentiment Analysis

The notebook considers the task of assigning sentiments ( positive, negative, or neutral) towards certain aspects of the input. Here the aspect term denotes the specific text that explicitly appears in the given text. The notebook shows results on a small subset of the dataset with customer reviews of laptops. Zero-shot and Few-shot tasks are also explored in this notebook. 

## Ensembling Example

In the folder `prompt_ensembling` is an example of how one might use prompt ensembling to improve the accuracy an LLMs performance on a downstream task. The idea is to combine prompts in order to ask the model to perform a task in multiple ways, hopefully producing better performance by asking the same question in multiple ways.

The notebook considers the Balanced Choice of Plausible Alternatives (CoPA) task, which is a harder version of the original CoPA task. We work with a small sample of the dataset and perform few-shot prompting of the vector LLMs. The task, in short, is, given a context and a premise of either cause or effect, the model must choose between two distinct sentences to determine which is the logical following sentence. An example is:

From the following choices,
1) The author faded into obscurity.
2) It was adapted into a movie.

and an effect premise, which logically follows the sentence "The book became a huge failure." The answer is "The author faded into obscurity."

We consider vote ensembling, averaging of label probabilities, and "bootstrap" ensembles. Then we determine whether the accuracy is improved after ensembling. In addition, we experiment with two separate scoring mechanisms for model responses. The first is Rouge scores for generated labels. The second is phrase likelihood as estimated by the model logits.

## Can we transfer gradient optimized prompts to OPT?

In the folder `transfering_gradient_optimized_prompts`, we consider whether the gradient optimized prompts that we found during discrete prompt optimization through gradient based search for the T5 model are transferrable to the OPT 6.7B model for sentiment analysis. This model is larger than T5, but has a different tokenizer and a different architecture. T5 has an encoder-decoder coupling, while OPT is decoder-only.
