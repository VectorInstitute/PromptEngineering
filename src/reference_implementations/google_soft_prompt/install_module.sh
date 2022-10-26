pip install \
	--upgrade "jax[cuda]" \
	-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

git clone --branch=main https://github.com/google-research/prompt-tuning
cd prompt-tuning
pip install . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd ..
