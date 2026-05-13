from solver import Solver
from config import *
import jax
from jax import numpy as jnp

"""
We can define additional problem classes in this file.
After creating one, make sure to register it in the 'Solver_Registry'.
"""

class HJB_Solver(Solver):
    
    def HJB_get_exact_X0(self):
        return jnp.zeros(self.model_config.d_in)
    
    def HJB_analytic_X_for_default_bc(self, T, W):
        return jnp.broadcast_to(self.HJB_get_exact_X0()[jnp.newaxis, jnp.newaxis, :], (T.shape[0], 1, self.model_config.d_in)) + jnp.sqrt(2.0)*W

    def HJB_analytic_u_for_default_bc(self, t, x):
        w = jnp.sqrt(self.solver_config.T-t) * jax.random.normal(jax.random.key(10), (100000, self.model_config.d_in))
        return -jnp.log(jnp.mean(jnp.exp(-self.bc_fn(x + jnp.sqrt(2)*w)), axis=0))

    def HJB_pinns_residual(self, params, t, x):
        _, ut, ux = self.calc_ut_ux(params, t, x)
        _, lap = self.calc_laplacian(params, t, x)
        loss = jnp.mean((ut[..., 0] + lap - jnp.sum(ux**2, axis=-1))**2)
        return loss
    
    def HJB_b(self, t, x):
        return jnp.zeros_like(x)

    def HJB_sigma(self, t, x):
        return jnp.sqrt(2) * jnp.broadcast_to(jnp.eye(x.shape[-1]), (*x.shape[:-1], x.shape[-1], x.shape[-1]))
    
    def HJB_h(self, t, x, y, z):
        return jnp.sum(z**2, axis=-1)

    def HJB_b_heun(self, t, x):
        return jnp.zeros_like(x)

    def HJB_c(self, t, x, u, ux, weighted_lap):
        return 0.5*weighted_lap



class BSB_Solver(Solver):

    def BSB_get_exact_X0(self):
        return jnp.concatenate([(jnp.ones(1) if i%2 == 0 else jnp.ones(1)/2) for i in range(self.model_config.d_in)])
    
    def BSB_analytic_X_for_default_bc(self, T, W):
        return jnp.broadcast_to(self.BSB_get_exact_X0()[jnp.newaxis, jnp.newaxis, :], (T.shape[0], 1, self.model_config.d_in)) * jnp.exp(0.4*W - 0.5*0.4**2*T)

    def BSB_analytic_u_for_default_bc(self, t, x):
        return jnp.exp((0.05 + 0.4**2)*(self.solver_config.T - t)) * self.bc_fn(x)
    
    def BSB_pinns_residual(self, params, t, x):
        u, ut, ux = self.calc_ut_ux(params, t, x)
        _, weighted_lap = self.calc_laplacian(params, t, x, weight=self.BSB_sigma(t, x))
        loss = jnp.mean((ut[..., 0] + 0.5*weighted_lap - 0.05*(u - jnp.matmul(ux, x[..., jnp.newaxis])[..., 0]))**2)
        return loss

    def BSB_b(self, t, x):
        return jnp.zeros_like(x)
    
    def BSB_sigma(self, t, x):
        return 0.4 * jax.vmap(jnp.diag, in_axes=0)(x)
    
    def BSB_h(self, t, x, y, z):
        return 0.05 * (y - jnp.matmul(z, x[..., jnp.newaxis])[..., 0])
    
    def BSB_b_heun(self, t, x):
        return -0.5 * 0.4**2 * x
    
    def BSB_c(self, t, x, u, ux, weighted_lap):
        return 0.5*weighted_lap + 0.5 * 0.4**2 * jnp.matmul(ux, x[..., jnp.newaxis])[..., 0]



class AC_Solver(Solver):

    def AC_get_exact_X0(self):
        return jnp.zeros(self.model_config.d_in)
    
    def AC_analytic_X_for_default_bc(self, T, W):
        return jnp.broadcast_to(self.AC_get_exact_X0()[jnp.newaxis, jnp.newaxis, :], (T.shape[0], 1, self.model_config.d_in)) + W
    
    def AC_analytic_u_for_default_bc(self, t, x):
        return jnp.full_like(self.bc_fn(x), 0.30879)  # for d_in = 20 and T = 0.3
    
    def AC_pinns_residual(self, params, t, x):
        u, ut, ux = self.calc_ut_ux(params, t, x)
        _, weighted_lap = self.calc_laplacian(params, t, x, weight=self.AC_sigma(t, x)) 

        residual = ut[..., 0] + 0.5 * weighted_lap + u - u**3
        loss = jnp.mean(residual**2)
        return loss

    def AC_b(self, t, x):
        return jnp.zeros_like(x)

    def AC_sigma(self, t, x):
        return jnp.broadcast_to(jnp.eye(x.shape[-1]), (*x.shape[:-1], x.shape[-1], x.shape[-1]))

    def AC_h(self, t, x, y, z):
        return - y + y**3

    def AC_b_heun(self, t, x):
        return jnp.zeros_like(x)

    def AC_c(self, t, x, u, ux, weighted_lap): 
        return 0.5 * weighted_lap


SOLVER_REGISTRY = {
    "HJB": HJB_Solver,
    "BSB": BSB_Solver,
    "AC":  AC_Solver,
}