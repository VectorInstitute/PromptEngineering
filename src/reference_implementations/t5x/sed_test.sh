#!/bin/bash
cd t5x
sed -i "s/_jax_version = '0.4.3'/_jax_version = '0.4.1'/" setup.py
sed -i "s/_jaxlib_version = '0.4.3'/_jaxlib_version = '0.4.1'/" setup.py
sed -i "s/flax @ git+https://github.com/google/flax#egg=flax/flax>=0.5.1" setup.py
sed -i "s/jax >=/jax ==/" setup.py
sed -i "s/jaxlib >=/jaxlib ==/" setup.py

