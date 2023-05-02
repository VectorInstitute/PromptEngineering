# Hands-On Exploration Guide

This guide provides some suggestions for areas of the repository that participants might explore based on the topics covered in preceding lecture days. Please note that these are simply suggestions and meant to help orient participants within the repository a bit better. As a participant, you need not follow this guide and should feel free to engage with any material in this repository that interests you. If it makes sense to dedicate your time completely to a single area of this repository, please do so to maximize your personal learning.

## Hands-On Day 1 (Lab Day 3)

In the lectures preceding Hands-On Day 1, we will have covered a number of topics including:

* Language Modeling and Evaluation Methods.
* Pre-training, fine-tuning, zero-shot and few-shot methods.
* Prompt Design, manual optimization, and ensembling.
* Challenges and capabilities associated with truly large language models.

As such, some areas of this respository that may be of interest are:

1. Hugging Face (HF) Basics: Fine-tuning NLP models using the HF API, using HF-hosted fine-tuned models for inference, and HF evaluation metrics.

    `src/reference_implementations/hugging_face_basics/`

2. Examples of Prompt design, manual optimization, zero- and few-shot prompting for various tasks. Tasks include classification, question answering, translation, summarization, and aspect-based sentiment analysis.

    `src/reference_implementations/prompting_vector_llms/llm_prompting_examples/`

    The README provides additional details for each of the examples in this folder.

    `src/reference_implementations/prompting_vector_llms/README.MD`

3. Some examples of ensembling multiple prompts to potentially improve performance.

    `src/reference_implementations/prompting_vector_llms/prompt_ensembling/`

## Hands-On Day 2 (Lab Day 5)

In the lectures preceding Hands-On Day 2, we will have covered several new topics including:

* Discrete and Continuous Prompt Optimization Techniques.
* Introduction to fairness/bias analysis for NLP models, including through prompting.
* Chain of Thought (CoT) prompting and vision prompts.

As such, some areas of this respository that could be of interest are:

1. An example of activation fine-tuning, with and without prompting.

    `src/reference_implementations/prompting_vector_llms/activation_fine_tuning/`

2. Examples of methods for fairness and bias analysis of LMs and LLMs

    `src/reference_implementations/fairness_measurement/`

    Notebooks for the BBQ and Crow-S pairs tasks are well document while the README in the `opt_czarnowska_analysis/` folder provides details about the code therein.

3. Examples of discrete prompt optimization ([AutoPrompt](https://arxiv.org/pdf/2010.15980.pdf), [GrIPS](https://arxiv.org/abs/2203.07281)).

    Source Code: `src/reference_implementations/prompt_zoo/`

    Associated Notebooks:

    `src/reference_implementations/prompt_zoo/experiment_notebooks/gradient_search_experiments.ipynb`

    `src/reference_implementations/prompt_zoo/experiment_notebooks/grips_experiments.ipynb`

4. Examples of continuous prompt optimization ([Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)).

    Source Code: `src/reference_implementations/prompt_zoo/`

    Experiment Instructions:

    `src/reference_implementations/prompt_zoo/experiment_notebooks/efficient_tuning_baselines.md`

5. While there are, presently, no notebooks that explicitly implement CoT prompting, we encourage you to try implementing CoT prompting concepts into queries of OPT either by modifying the notebooks in `src/reference_implementations/prompting_vector_llms/` or by creating your own.

## Hands-On Day 3 (Lab Day 7)

In the Frontiers Talks preceding Hands-On Day 3, additional parameter-efficient fine-tuning (PEFT) approaches will be covered, along with Augmented Retrieval methods.

On this third day, we encourage participants to continue exploring implementations and examples that interest them most, based on their previous days of investigation, within the repository.
