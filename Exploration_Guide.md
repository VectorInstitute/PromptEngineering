# Hands-On Exploration Guide

This guide provides some suggestions for areas of the repository that participants might explore based on the topics covered in preceding lecture days. Please note that these are simply suggestions and meant to help orient participants within the repository a bit better. As a participant, you need not follow this guide and should feel free to engage with any material in this repository that interests you. If it makes sense to dedicate your time completely to a single area of this repository, please do so to maximize your personal learning.

## Hands-On Day 1 (Lab Day 3)

In the lectures preceding Hands-On Day 1, we will have covered a number of topics including:

* Language Modeling and Evaluation Methods
* Pre-training, fine-tuning, zero-shot and few-shot methods
* Prompt Design, manual optimization, and ensembling
* Challenges and capabilities associated with truly large language models.

As such, some areas of this respository that could be of interest are:

1. Hugging Face (HF) Basics: Fine-tuning NLP models using the Hugging Face API, Using HF-hosted fine-tuned models for inference, and HF evaluation metrics.

    `src/reference_implementations/hugging_face_basics/`

2. Examples of Prompt Design, Optimization, Zero- and Few-shot Prompting for various tasks.

    `src/reference_implementations/prompting_vector_llms/llm_prompting_examples/`

    The README

    `src/reference_implementations/prompting_vector_llms/README.MD`

    provides additional details for each of the examples in this folder.

3. Some examples of Ensembling Multiple Prompts to potentially improve performance.

    `src/reference_implementations/prompting_vector_llms/prompt_ensembling/`

## Hands-On Day 2 (Lab Day 5)

In the lectures preceding Hands-On Day 2, we will have covered a number of topics including:

* Discrete and Continuous Prompt Optimization Techniques.
* Introduction to fairness/bias analysis for NLP models, including through prompting.
* Chain of Thought (CoT) Prompting and Vision Prompts

As such, some areas of this respository that could be of interest are:

1. An example of activation fine-tuning, with and without prompting.

    `src/reference_implementations/prompting_vector_llms/activation_fine_tuning/`

2. Examples of Methods for Fairness and Bias Analysis of LMs and LLMs

    `src/reference_implementations/fairness_measurement/`

    Notebooks for the BBQ and Crow-S Pairs tasks are well document while the README in the `opt_czarnowska_analysis` folder provides details about the code therein.

3. Examples of Discrete Prompt Optimization (AutoPrompt, GrIPS).

    Source Code: `src/reference_implementations/prompt_zoo/`

    Notebooks:

    `src/reference_implementations/prompt_zoo/experiment_notebooks/gradient_search_experiments.ipynb`

    `src/reference_implementations/prompt_zoo/experiment_notebooks/grips_experiments.ipynb`

4. Examples of Continuous Prompt Optimization (Prompt Tuning).

    Source Code: `src/reference_implementations/prompt_zoo/`

    Experiment Instructions:

    `src/reference_implementations/prompt_zoo/experiment_notebooks/efficient_tuning_baselines.md`

5. While there are no notebooks that explicitly implement CoT prompting, we encourage you to try CoT prompting concepts into queries of OPT.

## Hands-On Day 3 (Lab Day 7)

In the Frontiers Talks preceding Hands-On Day 2, additional parameter-efficient fine-tuning (PEFT) approaches will be covered, along with Augmented Retrieval methods.

On the third day, we encourage participants to continue exploring implementations and examples that interest them most, based on their previous days of investigation, within the repository.




    `
