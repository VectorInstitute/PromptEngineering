# A Discussion of Vector Hosted LLM Generation Configuration Parameters

In this README, we discuss, in a bit more detail, how the configuration parameters for Vector's LLMs affect the generation/output that they produce. While the configuration setup is specific to Vector, these parameters, in their various forms, are often used across many other generative LMs.

To start off, let's just list off all of the options available for configuration.

* `max_tokens` Sets the maximum number of tokens the model generates before haulting generation. Note that this does not guarantee the model concludes cleanly. For example, it may stop generating mid-thought if it hits this threshold.

* `top_k`: Range: 0 to model vocabulary size (50272 for OPT). At each generation step this is the number of tokens to select from with probabilities associated with their relative likelihoods. Setting this to 1 is "Greedy decoding." If `top_k` is set to zero, then we exclusively use nucleus sampling (i.e. `top_p` below).

    __NOTE__: Greedy decoding (top_k = 1) is often useful for obtaining factually correct responses, but increases the probability that the model becomes repetitive/circular in longer generations.

* `top_p`: Range: 0.0-1.0, also known as nucleus sampling. At each generation step, the tokens with the largest probabilities, adding up to `top_p` are sampled from relative to their likelihoods.

    __NOTE__: When both `top_k` and `top_p` are set, the more restrictive of the two is used. That is, if `top_p` would be selecting from 5 tokens and `top_k` selects from 6, 5 tokens are selected from in the generation step.

* `rep_penalty`: Range >= 1.0. This attempts to decrease the likelihood of tokens in a generation process if they have already been generated in the current trajectory. A value of 1.0 means no penalty and larger values increasingly penalize repeated values. A value of 1.2 has been reported as a good default.

* `temperature`: Range >=0.0. This value "sharpens" or "flattens" the softmax calculation done to produce probabilties over the vocabulary of the LM. A value of 1.0 collapses to the standard softmax calculation.

    * As temperature goes to zero: only the largest probabilities will remain non-zero (approaches greedy decoding).

    * As it approaches infinity, the distribution spreads out evenly over the vocabulary.

    __NOTE__ To increase sampling diversity (possibly at the expense of sensical generation) increase temperature. To produce less diversity, decrease it.

* `n`: An integer value of 1 or greater. This is the number of beams to be used in beam search for generative decoding. There are a lot of great articles online discussing beam search if you are unfamiliar with the decoding concept. The __default is 1__. The more beams that you request, the longer it will take for the model to respond, but the fluency of the response may be improved. It is also fairly memory intensive, so please do not set this too high if you choose to use multiple beams.

* `stop`: This can be a single string or a list of strings. These are values that tell the model when to stop decoding. For example, it could simply be a "." to tell the model it is done generating after producing a period.

### Existing Resources For Additional Details

There are already several detailed discussions of how some of these parameters affect the generation of langauge models. Two useful blog posts are:

* [Decoding Strategies that You Need to Know for Response Generation](https://txt.cohere.ai/llm-parameters-best-outputs-language-ai/).
* [LLM Parameters Demystified: Getting The Best Outputs from Language AI](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc).

These links cover the basics around parameters like `top_k`, `top_p`, `temperature`, and `stop` words. There are some nice illustration of how these concepts manipulate the generative steps. They also touch briefly on beam search and repetition penalty. However, we find that the most comprehensive and clear description for `rep_penalty` is actually in the paper that introduce it. That paper is:

* [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)

For each task that you are working on, it is important to experiment with these parameters to get the best generative results possible. As mentioned above, some settings are particularly good for factually correct responses, while others might induce better fluency. Stop words can help you trim responses without significant post-processing procedures, but may also short-circuit potentially helpful longer generations.
