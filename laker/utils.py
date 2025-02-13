import jax
import jax.numpy as jnp

def spectral_norm(w):
    # two-sided power iteration
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, w.shape[1])
    
    def power_iteration_step(v, _):
        v = w @ v
        v = v / jnp.linalg.norm(v)
        v = w.T @ v
        v = v / jnp.linalg.norm(v)
        return v, None
    
    v, _ = jax.lax.scan(power_iteration_step, v, None, length=8)
    return jnp.linalg.norm(w @ v)

