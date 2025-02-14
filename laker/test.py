import jax
import jax.numpy as jnp

input_dim = 64
output_dim = 16
batch_size = 128

key = jax.random.PRNGKey(0)
inputs = jax.random.normal(key, (input_dim, batch_size))
targets = jax.random.normal(key, (output_dim, batch_size))

from modula.abstract import Identity, Mul
from modula.atom import Linear
from modula.bond import ReLU

width = 32
depth = 4
lipschitz_constant = 4

linear = False

mlp = Linear(output_dim, width)
for _ in range(depth):
    if linear:
        mlp @= Linear(width, width)
    else:
        mlp @= Linear(width, width) @ ReLU()
mlp @= Linear(width, input_dim)
mlp @= Mul(lipschitz_constant)

print(mlp)

mlp.jit()

from modula.error import SquareError

error = SquareError()

from modula.atom import orthogonalize

steps = 200
learning_rate = 0.5 / lipschitz_constant
weight_decay = 0.00

report_steps = 20
feature_learning = []
weight_norms = []

key = jax.random.PRNGKey(0)
w = mlp.initialize(key)

for step in range(steps):
    # compute outputs and activations
    outputs, activations = mlp(inputs, w)
    
    # compute loss
    loss = error(outputs, targets)
    
    # compute error gradient
    error_grad = error.grad(outputs, targets)
    
    # compute gradient of weights
    grad_w, _ = mlp.backward(w, activations, error_grad)
    
    # dualize gradient
    d_w = mlp.dualize(grad_w)

    # compute scheduled learning rate
    lr = learning_rate# * (1 - step / steps)

    # weight decay
    # w = [weight * (1 - lr * weight_decay) for weight in w]
    
    # update weights
    w = [weight - lr * d_weight for weight, d_weight in zip(w, d_w)]

    # orthogonalize weights
    w = mlp.project(w)

    if step % report_steps == 0:
        print(f"Step {step:3d} \t Loss {loss:.6f}")
        #singular_values_d_w = [jnp.linalg.svd(d_weight, compute_uv=False) for d_weight in grad_w]
        grad_norms = [jnp.linalg.norm(d_weight) for d_weight in grad_w]
        feature_learning.append(grad_norms)
        #feature_learning.append([s[0] for s in singular_values_d_w])
        singular_values_w = [jnp.linalg.svd(wi, compute_uv=False) for wi in w]
        weight_norms.append([(s[0], s[-1]) for s in singular_values_w])

import matplotlib.pyplot as plt

# Create gif of feature learning and weight norms over time
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
feature_learning = jnp.array(feature_learning)
weight_norms = jnp.array(weight_norms)

def update(frame):
    ax1.clear()
    ax2.clear()
    
    # Plot feature learning on left y-axis (blue)
    ax1.plot(range(len(w)), feature_learning[frame], '-o', color='blue')
    ax1.set_xlabel('Layer Depth')
    ax1.set_ylabel('Feature Learning Strength (Gradient Norm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot weight matrix singular value ranges on right y-axis (red)
    layer_indices = range(len(w))
    max_singular_values = weight_norms[frame, :, 0]  # Largest singular values
    min_singular_values = weight_norms[frame, :, 1]  # Smallest singular values
    mean_singular_values = (max_singular_values + min_singular_values) / 2
    singular_value_ranges = max_singular_values - min_singular_values
    
    ax2.errorbar(layer_indices, mean_singular_values, 
                yerr=singular_value_ranges/2,  # Divide by 2 since errorbar expects symmetric errors
                fmt='o', color='red', capsize=5, 
                label='Singular Value Range')
    ax2.set_ylabel('Weight Matrix Singular Values', color='red', rotation=270, labelpad=15)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.set_label_position('right')
    
    plt.title(f'Learning Dynamics at Step {frame * report_steps}\n{"Linear" if linear else "ReLU"} Network, width {width}, depth {depth}')
    ax1.grid(True)
    
    # Set consistent y-axis scales
    ax1.set_ylim(0, feature_learning.max() * 1.1)
    ax2.set_ylim(0, weight_norms[:, :, 0].max() * 1.1)  # Use max of largest singular values
    plt.tight_layout()

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, update, frames=len(feature_learning), interval=200)
anim.save('learning_dynamics.gif', writer='pillow')
plt.close()

