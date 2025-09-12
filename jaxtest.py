'''
GPU JAX verify test
general test file for jaxopt projects
'''

import jax
import jax.numpy as jnp
from jax.extend.backend import get_backend

print(get_backend().platform)

# JAX version should be 0.4.33
print("Running JAX Version =",jax.__version__)


# Do a simple array computation on GPU
array = jnp.array([0,1,2,3,4])
print('Adding the array')
print('Done')

