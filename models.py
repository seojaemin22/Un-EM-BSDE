import jax.numpy as jnp
import flax.linen as nn
import jax
from config import *
from jax.nn import initializers
from abc import ABC, abstractmethod
from typing import Optional, Callable


def get_model(model_config: Model_Config, problem_name):

    if model_config.model_name == 'MLP':
        activation = ACTIVATION_REGISTRY[model_config.MLP_activation]()
        boundary_function = BC_REGISTRY[f"{problem_name}_{model_config.bc_name}"]
        boundary_laplacian_function = BC_LAPLACIAN_REGISTRY[f"{problem_name}_{model_config.bc_name}"]
        return MLP(model_config, activation=activation, boundary_function=boundary_function, boundary_laplacian_function=boundary_laplacian_function)
    else:
        raise Exception("Model '" + model_config.model_name + "' Not Implemented")

# --------------------------------------------------

class Activation(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def deriv1(self, x):
        pass

    @abstractmethod
    def deriv2(self, x):
        pass


class Sin(Activation):

    def __call__(self, x):
        return jnp.sin(x)
    
    def deriv1(self, x):
        return jnp.cos(x)
    
    def deriv2(self, x):
        return -jnp.sin(x)
    
class Tanh(Activation):

    def __call__(self, x):
        return jnp.tanh(x)
    
    def deriv1(self, x):
        return 1.0 - jnp.tanh(x)**2
    
    def deriv2(self, x):
        return -2.0 * jnp.tanh(x) * (1.0 - jnp.tanh(x)**2)

class Mish(Activation):

    def __call__(self, x):
        return x * jnp.tanh(nn.softplus(x))
    
    def deriv1(self, x):
        sp = nn.softplus(x)
        tsp = jnp.tanh(sp)
        return tsp + x * (1.0 - tsp**2) * jax.nn.sigmoid(x)
    
    def deriv2(self, x):
        sp = nn.softplus(x)
        tsp = jnp.tanh(sp)
        dsp = jax.nn.sigmoid(x)
        dtsp = (1.0 - tsp**2) * dsp
        d2tsp = -2.0 * tsp * dtsp * dsp + (1.0 - tsp**2) * dsp * (1.0 - dsp)
        return 2.0 * dtsp + x * d2tsp

class ReLU(Activation):
    def __call__(self, x):
        return jnp.maximum(x, 0.0)
    
    def deriv1(self, x):
        return jnp.where(x > 0.0, 1.0, 0.0)
    
    def deriv2(self, x):
        return jnp.zeros_like(x)

class LeakyReLU(Activation):
    def __call__(self, x):
        return jnp.maximum(x, 0.01*x)
    
    def deriv1(self, x):
        return jnp.where(x > 0.0, 1.0, 0.01)
    
    def deriv2(self, x):
        return jnp.zeros_like(x)
    

ACTIVATION_REGISTRY = {
    'sin': Sin,
    'tanh': Tanh,
    'mish': Mish,
    'relu': ReLU,
    'leakyrelu': LeakyReLU,
}

# --------------------------------------------------

def HJB_default_bc(x):
    return jnp.log(0.5 * (1 + jnp.sum(x**2, keepdims=True, axis=-1)))

def BSB_default_bc(x):
    return jnp.sum(x**2, keepdims=True, axis=-1)

def AC_default_bc(x): 
    norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)
    return 1.0 / (2.0 + 0.4 * norm_sq)


BC_REGISTRY = {
    'HJB_default': HJB_default_bc,
    'BSB_default': BSB_default_bc,
    "AC_default": AC_default_bc
}

# --------------------------------------------------

def _trM_and_xMx(x, weight=None):

    if weight is None:
        trM = x.shape[-1]  # d
        xMx = jnp.sum(x * x, axis=-1, keepdims=True)
        trM = jnp.asarray(trM, dtype=x.dtype)
        return trM, xMx

    W = weight
    trM = jnp.sum(W * W, axis=(-2, -1))
    trM = trM[..., None]
    y = jnp.einsum('...i,...iD->...D', x, W)
    xMx = jnp.sum(y * y, axis=-1, keepdims=True)
    return trM, xMx


def laplacian_HJB_default(x, weight=None):
    trM, xMx = _trM_and_xMx(x, weight)
    r2 = jnp.sum(x * x, axis=-1, keepdims=True)
    s = 1.0 + r2
    return (2.0 / s) * trM - (4.0 / (s * s)) * xMx

def laplacian_BSB_default(x, weight=None):
    trM, _ = _trM_and_xMx(x, weight)
    return 2.0 * trM

def laplacian_AC_default(x, weight=None):
    trM, xMx = _trM_and_xMx(x, weight)
    r2 = jnp.sum(x * x, axis=-1, keepdims=True)
    a, b = 2.0, 0.4
    s = a + b * r2
    return (-2.0 * b / (s * s)) * trM + (8.0 * b * b / (s * s * s)) * xMx

BC_LAPLACIAN_REGISTRY = {
    'HJB_default': laplacian_HJB_default,
    'BSB_default': laplacian_BSB_default,
    'AC_default':  laplacian_AC_default,
}

# --------------------------------------------------

class MLP(nn.Module):
    config: Model_Config
    activation: Activation
    boundary_function: Optional[Callable] = None
    boundary_laplacian_function: Optional[Callable] = None

    def setup(self):
        param_dtype = jnp.float64 if self.config.use_float64 else jnp.float32
        zero_init   = initializers.zeros

        if self.config.MLP_kernel_init == 'he':
            he_init     = nn.initializers.variance_scaling(2.0, 'fan_in', 'truncated_normal')
            self.layers = [nn.Dense(self.config.MLP_d_hidden, kernel_init=he_init, bias_init=zero_init, param_dtype=param_dtype) for _ in range(self.config.MLP_num_layers)]
            self.output_layer = nn.Dense(self.config.d_out, kernel_init=he_init, bias_init=zero_init, param_dtype=param_dtype)
        else:  # self.config.MLP_kernel_init == 'xavier'
            xavier_init = initializers.glorot_uniform()
            self.layers = [nn.Dense(self.config.MLP_d_hidden, kernel_init=xavier_init, bias_init=zero_init, param_dtype=param_dtype) for _ in range(self.config.MLP_num_layers)]
            self.output_layer = nn.Dense(self.config.d_out, kernel_init=xavier_init, bias_init=zero_init, param_dtype=param_dtype)

    def __call__(self, *args):
        src = jnp.concatenate(args, axis=-1)
        if self.config.MLP_skip_conn:
            src_skip = jnp.zeros((src.shape[0], self.config.MLP_d_hidden))
            
        for i in range(len(self.layers)):
            src = self.layers[i](src)
            src = self.activation(src)
            if self.config.MLP_skip_conn:
                if i in self.config.MLP_save_layers:
                    src_skip = src
                if i in self.config.MLP_skip_layers:
                    src = src + src_skip

        src = self.output_layer(src)

        if self.config.use_hard_constraint:
            t, x = args
            boundary_value = self.boundary_function(x)
            return boundary_value + (self.config.T - t) * src
        else:
            return src

    # --------------------------------------------------
    
    def forward_laplacian(self, params, *args, weight=None):
        src = jnp.concatenate(args, axis=-1)
        B = src.shape[0]
        Din_total = src.shape[1]
        Din = self.config.d_in

        G = jnp.zeros((B, Din_total, Din))
        G = G.at[:, 1:, :].set(jnp.eye(Din) if weight is None else weight)  # (weighted) gradient
        L = jnp.zeros((B, Din_total))  # (weighted) laplacian

        if self.config.MLP_skip_conn:
            src_skip = jnp.zeros((B, self.config.MLP_d_hidden))
            G_skip   = jnp.zeros((B, self.config.MLP_d_hidden, Din))
            L_skip   = jnp.zeros((B, self.config.MLP_d_hidden))

        for i in range(len(params['params']) - 1):
            W = params['params'][f'layers_{i}']['kernel']  # (Hin, Hout)
            b = params['params'][f'layers_{i}']['bias']    # (Hout,)

            # Linear layer : z(v) = L(x(v)) = W x(v) + b
            # dz(v) = W dx(v)
            # d2z(v) = W d2x(v)
            # lap(z(v)) = W lap(x(v))

            z = jnp.einsum('...i,ih->...h', src, W) + b  # (B, Hout)
            G = jnp.einsum('...jD,jh->...hD', G, W)      # (B, Hout, Din)
            L = jnp.einsum('...j,jh->...h', L, W)        # (B, Hout)

            # Activation layer : z(v) = phi(x(v))
            # dz(v) = phi'(x(v)) dx(v)
            # d2z(v) = phi''(x(v)) d2x(v) + phi'(x(v)) (dx(v))^2
            # lap(z(v)) = phi''(x(v)) lap(z(v)) + phi'(x(v)) sum (dx(v))^2

            phi1 = self.activation.deriv1(z)  # (B, Hout)
            phi2 = self.activation.deriv2(z)  # (B, Hout)

            L = phi2 * jnp.sum(G**2, axis=-1) + phi1 * L
            G = phi1[..., None] * G
            src = self.activation(z)

            # skip connection
            if self.config.MLP_skip_conn:
                if i in self.config.MLP_save_layers:
                    src_skip, G_skip, L_skip = src, G, L
                if i in self.config.MLP_skip_layers:
                    src = src + src_skip
                    G = G + G_skip
                    L = L + L_skip

        # output layer
        W = params['params']['output_layer']['kernel']
        b = params['params']['output_layer']['bias']
        src = jnp.einsum('...h,hk->...k', src, W) + b
        G = jnp.einsum('...hD,hk->...kD', G, W)
        L = jnp.einsum('...h,hk->...k',  L, W)

        u_raw = src[..., :self.config.d_out]
        lap_raw = L[..., :self.config.d_out]

        if self.config.use_hard_constraint:
            t, x = args
            boundary_value = self.boundary_function(x)
            boundary_laplacian = self.boundary_laplacian_function(x, weight)
            u = boundary_value + (self.config.T - t) * u_raw
            lap = boundary_laplacian + (self.config.T - t) * lap_raw
            return u, lap
            
        return u_raw, lap_raw