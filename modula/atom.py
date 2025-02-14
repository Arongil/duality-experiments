import jax
import jax.numpy as jnp

from modula.abstract import Atom

def orthogonalize(M):
    a, b, c = 3.0, -3.2, 1.2
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for _ in range(10):
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


class Linear(Atom):
    def __init__(self, fanout, fanin, ortho_backwards=False, residual_init=False):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
        self.scale = (self.fanout / self.fanin) ** 0.5
        self.ortho_backwards = ortho_backwards
        self.residual_init = residual_init

    def forward(self, x, w):
        weights = w[0]
        return weights @ x, [x] #* self.scale, [x]

    def backward(self, w, acts, grad_output):
        weights = w[0]
        input = acts[0]
        # idea 1: renormalize grad_input every time to be unit vector (it'll be dualized away anyway -- just to prevent gradients from deviating from unit norm)
        # idea 2: pretend like this linear layer is orthogonal when flowing gradient backwards
        if self.ortho_backwards:
            grad_input = self.project([weights.T])[0] @ grad_output
        else:
            grad_input = weights.T @ grad_output
        grad_weight = grad_output @ input.T
        #print(f"grad_output: {grad_output.shape}")
        #print(f"weights: {weights.shape}, input: {input.shape}")
        #print(f"grad_weight: {grad_weight.shape} with avg norm {jnp.mean(jnp.linalg.norm(grad_weight)).item()}")
        #print(f"grad_input: {grad_input.shape} with avg norm {jnp.mean(jnp.linalg.norm(grad_input)).item()}")
        return [grad_weight], grad_input

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        projected = self.project([weight])[0]
        if self.residual_init and self.fanout == self.fanin:
            projected = projected + jnp.eye(*projected.shape)  # simulate residual connection via initialization
        return [projected]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
        return [d_weight]


class Embed(Atom):
    def __init__(self, d_embed,num_embed):
        super().__init__()
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]
        return weights[:, x], [x]

    def backward(self, w, acts, grad_output):
        weights = w[0]
        x = acts[0]
        grad_input = None
        grad_weight = jnp.zeros_like(weights)
        grad_weight = grad_weight.at[:, x].add(grad_output[:,x])
        return [grad_weight], grad_input

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.d_embed, self.num_embed))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight, axis=0, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    # sample a random d0xd1 matrix
    d0, d1 = 50, 100
    M = jax.random.normal(key, shape=(d0, d1))
    O = orthogonalize(M)

    # compute SVD of M and O
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    s = jnp.linalg.svd(O, compute_uv=False)

    # print singular values
    print(f"min singular value of O: {jnp.min(s)}")
    print(f"max singular value of O: {jnp.max(s)}")

    # check that M is close to its SVD
    error_M = jnp.linalg.norm(M - U @ jnp.diag(S) @ Vh) / jnp.linalg.norm(M)
    error_O = jnp.linalg.norm(O - U @ Vh) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in M's SVD: {error_M}")
    print(f"relative error in O: {error_O}")
