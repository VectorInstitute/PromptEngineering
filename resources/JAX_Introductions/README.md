# Resources
    # Links related to XLA
        -   XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.
        -	https://www.tensorflow.org/xla
        -	https://www.tensorflow.org/xla/architecture
        -	https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html
        -	https://www.tensorflow.org/xla/tutorials/jit_compile

    # Understand more about the Autograd
        -   Autograd can automatically differentiate native Python and Numpy code.
        - 	https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
        -	Some examples:
            - https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
            - https://github.com/HIPS/autograd/blob/master/examples/rnn.py
            - https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf

    # Reading Jax
        -   JAX is Autograd and XLA, brought together for high-performance machine learning research.
        -   Check the notebook 'you-don-t-know-jax.ipynb'
        -	Main documents: [https://github.com/google/jax, https://jax.readthedocs.io/en/latest/]
        -   Common Gotchas in Jax:
                https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb#scrollTo=w99WXa6bBa_H

    # Reading Flax
        -   Flax is a high-performance neural network library and ecosystem for JAX that is designed for flexibility.
        -   Main Repo: https://github.com/google/flax
        -   Check this MNIST classifier using Flax:
            https://colab.research.google.com/github/google/flax/blob/main/docs/getting_started.ipynb#scrollTo=7ipyJ-JGCNqP

    # Reading XManager
        -   XManager is a platform for packaging, running and keeping track of machine learning experiments. It currently enables one to launch experiments locally or on Google Cloud Platform (GCP).
        -   https://storage.googleapis.com/gresearch/xmanager/deepmind_xmanager_slides.pdf
        -   https://github.com/deepmind/xmanager

    # Reading gin-config
        -   https://github.com/google/gin-config
        -   https://github.com/google/gin-config/blob/master/docs/index.md

    # Reading seqio
        -   Used to preprocess or postprocess the data for sequence models.
        -   https://github.com/google/seqio

    # Reading FlaxFormer
        -   Used to implement the T5 and other architectures based on jax and flax.
        -   https://github.com/google/flaxformer/blob/main/flaxformer/architectures/t5/t5_1_1.py
        -   https://github.com/google/flaxformer/
        -   https://github.com/google/flaxformer/blob/main/flaxformer/t5x/configs/t5/models/t5_1_1_base.gin
        -   https://github.com/google/flaxformer/blob/main/flaxformer/t5x/configs/t5/architectures/t5_1_1_flaxformer.gin

    # Reading T5x
        -   Use the example configs.
        -   https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1
