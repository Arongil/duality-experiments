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
from modula.error import SquareError
from dataclasses import dataclass

@dataclass
class Config:
    width: int
    depth: int
    linear: bool
    residual: bool
    residual_init: bool
    lipschitz_constant: float
    lr: float
    steps: int
    report_steps: int
    weight_decay: float
    adam: bool
    momentum: float
    momentum2: float
    dualize_pre: bool
    dualize_post: bool
    project: bool
    ortho_backwards: bool
    make_learning_dynamics_plots: bool

def create_mlp(config: Config):
    width, depth, residual, linear, ortho_backwards, lipschitz_constant, residual_init = config.width, config.depth, config.residual, config.linear, config.ortho_backwards, config.lipschitz_constant, config.residual_init
    kwargs = {"ortho_backwards": ortho_backwards, "residual_init": residual_init}
    nonlinearity = Identity() if linear else ReLU()
    embed = Linear(width, input_dim, **kwargs)
    block = Linear(width, width, **kwargs) @ nonlinearity
    if residual:
        block = Identity() + block @ Linear(width, width, **kwargs)
    unembed = Linear(output_dim, width, **kwargs)
    lipschitz = Mul(lipschitz_constant)
    mlp = lipschitz @ unembed @ block ** depth @ embed
    mlp.jit()
    error = SquareError()
    return mlp, error

def train(config):
    key = jax.random.PRNGKey(0)
    mlp, error = create_mlp(config)
    w = mlp.initialize(key)

    lr = config.lr
    steps = config.steps
    report_steps = config.report_steps
    weight_decay = config.weight_decay
    adam = config.adam
    momentum = config.momentum
    momentum2 = config.momentum2
    dualize_pre = config.dualize_pre
    dualize_post = config.dualize_post
    project = config.project

    losses = []
    feature_learning = []
    weight_norms = []

    print(f"Training with lr {lr:.4f} for {steps} steps")

    m = [0 * weight for weight in w]
    m2 = [0 * weight for weight in w]
    for step in range(steps):
        outputs, activations = mlp(inputs, w)
        loss = error(outputs, targets)
        error_grad = error.grad(outputs, targets)
        grad_w, _ = mlp.backward(w, activations, error_grad)

        # pre_dualize, update first moment, update second moment, possibly apply adam, post_dualize
        d_m = mlp.dualize(grad_w) if dualize_pre else grad_w
        m = [momentum * m + (1-momentum) * d_m for m, d_m in zip(m, d_m)]
        m2 = [momentum2 * m2 + (1-momentum2) * d_m**2 for m2, d_m in zip(m2, d_m)]
        d_w = [m / (jnp.sqrt(m2) + 1e-8) if adam else m for m, m2 in zip(m, m2)]
        d_w = mlp.dualize(d_w) if dualize_post else d_w
        w = [weight - lr * d_weight for weight, d_weight in zip(w, d_w)]

        if weight_decay > 0:
            # this weight decay should really be applied before the weight update
            w = [weight * (1 - lr * weight_decay) for weight in w]
        
        if project:
            w = mlp.project(w)

        if step % report_steps == 0:
            print(f"Step {step:3d} \t Loss {loss:.6f}")
            losses.append(loss)
            grad_norms = [jnp.linalg.norm(d_weight) for d_weight in grad_w]
            feature_learning.append(grad_norms)
            #singular_values_d_w = [jnp.linalg.svd(d_weight, compute_uv=False) for d_weight in grad_w]
            #feature_learning.append([s[0] for s in singular_values_d_w])
            singular_values_w = [jnp.linalg.svd(wi, compute_uv=False) for wi in w]
            weight_norms.append([(s[0], s[-1]) for s in singular_values_w])
            
    return losses, feature_learning, weight_norms

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_learning_dynamics(config, feature_learning, weight_norms, title_prefix=""):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    feature_learning = jnp.array(feature_learning)
    weight_norms = jnp.array(weight_norms)
    num_layers = feature_learning.shape[-1]

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        ax1.plot(range(num_layers), feature_learning[frame], '-o', color='blue')
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Feature Learning Strength (Gradient Norm)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        layer_indices = range(num_layers)
        max_singular_values = weight_norms[frame, :, 0]
        min_singular_values = weight_norms[frame, :, 1]
        mean_singular_values = (max_singular_values + min_singular_values) / 2
        singular_value_ranges = max_singular_values - min_singular_values
        
        ax2.errorbar(layer_indices, mean_singular_values, 
                    yerr=singular_value_ranges/2,
                    fmt='o', color='red', capsize=5, 
                    label='Singular Value Range')
        ax2.set_ylabel('Weight Matrix Singular Values', color='red', rotation=270, labelpad=15)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_label_position('right')
        
        plt.title(f'{title_prefix}Learning Dynamics at Step {frame * config.report_steps}\n{"Linear" if config.linear else "ReLU"} Network, width {config.width}, depth {config.depth}')
        ax1.grid(True)
        
        ax1.set_ylim(0, feature_learning.max() * 1.1)
        ax2.set_ylim(0, weight_norms[:, :, 0].max() * 1.1)
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(feature_learning), interval=200)
    return anim

### SWEEP! ###

config = Config(
    width = 32,
    depth = 4,
    linear = False,             # whether to use any nonlinearity
    residual = False,           # whether to use residual connections
    residual_init = True,       # simulates residual connection via initialization I + W (best used separately from residual = True)
    lipschitz_constant = 4,     # final multiplicative factor
    lr = 0.01,
    steps = 200,
    report_steps = 20,          # how often to log progress
    weight_decay = 0.00,
    adam = True,                # transform using Adam moments (won't actually be Adam unless dualize_pre = dualize_post = False)
    momentum = 0.9,             # coefficient for first moment buffer
    momentum2 = 0.99,           # coefficient for second moment buffer
    dualize_pre = False,        # dualize gradient before inserting into momentum buffers
    dualize_post = False,       # dualize momentum buffers to create the final weight update
    project = True,             # project weights to be orthogonal after every step
    ortho_backwards = False,    # pretend like the weights are orthogonal in backprop
    make_learning_dynamics_plots = False,
)

lrs = jnp.logspace(-2.5, 0.5, 10)
final_losses = []

for lr in lrs:
    config.lr = lr
    losses, fl, wn = train(config)
    final_losses.append(losses[-1])
    
    # Save animation for this learning rate
    if config.make_learning_dynamics_plots:
        anim = plot_learning_dynamics(config, fl, wn, title_prefix=f"LR={lr:.1e} ")
        anim.save(f'laker/plots/learning_dynamics_lr_{lr:.1e}.gif', writer='pillow')
        plt.close()

# Plot learning rate sweep results
plt.figure(figsize=(10, 6))
plt.semilogx(lrs, final_losses, '-o')
plt.yscale('log')
plt.ylim(1e-3, 1e0)
plt.xlabel('Learning Rate')
plt.ylabel('Final Training Loss')
plt.title('Loss by Learning Rate')
plt.grid(True)
plt.savefig('laker/plots/lr_sweep.png')
plt.close()