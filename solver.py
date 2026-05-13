import jax
import flax.serialization as fs
import optax
import json
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
import tqdm
import wandb
import matplotlib.pyplot as plt
from models import *
from config import *
from functools import partial
import copy
from pathlib import Path

class Solver():

    def __init__(self, model_config: Model_Config, solver_config: Solver_Config):
        self.model_config = model_config
        self.solver_config = solver_config
        
        self.model = get_model(self.model_config, self.solver_config.problem_name)
        self.grad_setting()
        self.problem_setting()
        self.grad_fn = self.loss_to_grad(self.get_loss())
        self.optimizer = self.create_opt()

        self.sol_T, self.sol_X, self.sol_U = self.get_analytic_sol()
        if self.solver_config.save_to_wandb:
            self.init_wandb()
        else:
            self.save_dir = Path(self.solver_config.project_name) / self.solver_config.run_name
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.save_dir / 'log.txt', 'w')


    def grad_setting(self):

        def bind_apply(f):
            apply_fn = self.model.apply
            def wrapped(*args, **kwargs):
                return f(apply_fn, *args, **kwargs)
            return wrapped

        def maybe_ckpt(f):
            return jax.checkpoint(f) if self.solver_config.checkpointing else f

        self.calc_u      = maybe_ckpt(bind_apply(self._calc_u))
        self.calc_ut     = maybe_ckpt(bind_apply(self._calc_ut))
        self.calc_ux     = maybe_ckpt(bind_apply(self._calc_ux))
        self.calc_ut_ux  = maybe_ckpt(bind_apply(self._calc_ut_ux))
        self.calc_uxx    = maybe_ckpt(bind_apply(self._calc_uxx))

        if self.model_config.laplacian_method == 'forward':
            self.calc_laplacian = maybe_ckpt(bind_apply(self._forward_laplacian))
        elif self.model_config.laplacian_method == 'backward':
            self.calc_laplacian = maybe_ckpt(bind_apply(self._calc_laplacian))
    

    def problem_setting(self):
        problem_name = self.solver_config.problem_name
        bc_name = self.model_config.bc_name

        self.bc_fn = BC_REGISTRY[f"{problem_name}_{bc_name}"]

        self.get_exact_X0 = getattr(self, f'{problem_name}_get_exact_X0')

        self.analytic_X = getattr(self, f'{problem_name}_analytic_X_for_{bc_name}_bc')
        self.analytic_u = getattr(self, f'{problem_name}_analytic_u_for_{bc_name}_bc')
        
        self.pinns_residual = getattr(self, f'{problem_name}_pinns_residual')

        self.b = getattr(self, f'{problem_name}_b')
        self.sigma = getattr(self, f'{problem_name}_sigma')
        self.h = getattr(self, f'{problem_name}_h')

        self.b_heun = getattr(self, f'{problem_name}_b_heun')
        self.c = getattr(self, f'{problem_name}_c')
    
    def get_X0(self, key, batch):
        key, sub = jax.random.split(key)
        X0 = jnp.repeat(self.get_exact_X0()[None, ...], repeats=batch, axis=0) + self.solver_config.X0_std * jax.random.normal(sub, (batch, self.model_config.d_in))
        return key, X0


    def get_loss(self):
        loss_method = self.solver_config.loss_method
        if loss_method == 'FSPINNs': return self.FSPINNs_loss
        elif loss_method == 'EMBSDE': return self.EMBSDE_loss
        elif loss_method == 'HeunBSDE': return self.HeunBSDE_loss
        elif loss_method == 'MultiShotEMBSDE': return self.MultiShotEMBSDE_loss
        elif loss_method == 'UnEMBSDE': return self.UnEMBSDE_loss
        elif loss_method == 'Shotgun': return self.Shotgun_loss
        elif loss_method == 'UnShotgun': return self.UnShotgun_loss
        else: raise Exception("Loss Method '" + loss_method + "' Not Implemented")


    def create_opt(self):
        if self.solver_config.schedule == 'piecewise_constant':
            schedule = optax.piecewise_constant_schedule(
                init_value=self.solver_config.lr,
                boundaries_and_scales=self.solver_config.boundaries_and_scales
            )
        elif self.solver_config.schedule == 'cosine_decay':
            schedule = optax.cosine_decay_schedule(
                init_value=self.solver_config.lr,
                decay_steps=self.solver_config.iter
            )
        elif self.solver_config.schedule == 'cosine_onecycle':
            schedule = optax.cosine_onecycle_schedule(
                transition_steps=self.solver_config.iter,
                peak_value=self.solver_config.lr
            )
        else: # No schedule
            schedule = optax.constant_schedule(
                value=self.solver_config.lr
            )
            
        if self.solver_config.optim == 'adam':
            return optax.adam(learning_rate=schedule)
        elif self.solver_config.optim == 'adamw':
            return optax.adamw(learning_rate=schedule)
        else: # SGD
            return optax.sgd(learning_rate=schedule)
    

    def init_model(self, key):
        t_pde = jnp.zeros((self.solver_config.micro_batch, 1))
        x_pde = jnp.zeros((self.solver_config.micro_batch, self.model_config.d_in))
        key, sub = jax.random.split(key)
        return key, self.model.init(sub, t_pde, x_pde)

    def init_opt(self,params):
        return self.optimizer.init(params)
    

    def cast_tree_dtype(self, tree, dtype):
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype) 
                                      if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) 
                                      else x,
                                      tree)

    def init_solver(self, key):
        key, params = self.init_model(key)
        opt_state = self.init_opt(params)
        
        model_path = Path(self.solver_config.model_state)
        if model_path.exists():
            model_bytes = model_path.read_bytes()
            loaded_params = fs.from_bytes(params, model_bytes)

            if self.model_config.use_float64:
                params = self.cast_tree_dtype(loaded_params, jnp.float64)
            else:
                params = self.cast_tree_dtype(loaded_params, jnp.float32)
        
        opt_path = Path(self.solver_config.opt_state)
        if opt_path.exists():
            opt_bytes = opt_path.read_bytes()
            loaded_opt_state = fs.from_bytes(opt_state, opt_bytes)

            if self.model_config.use_float64:
                opt_state = self.cast_tree_dtype(loaded_opt_state, jnp.float64)
            else:
                opt_state = self.cast_tree_dtype(loaded_opt_state, jnp.float32)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if self.solver_config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return key, params, opt_state


    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------
    
    def calc_bcx(self, x): 
        jax_x = jax.jacrev(lambda x: self.bc_fn(x), argnums=0)
        return jax.vmap(jax_x, in_axes=0)(x)

    # --------------------------------------------------

    def _calc_u(self, apply_fn, params, t, x):
        return apply_fn(params, t, x)
    
    def _calc_ut(self, apply_fn, params, t, x):
        def u_ut(t, x):
            model_fn = lambda tt: self._calc_u(apply_fn, params, tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(u_ut, in_axes=(0, 0))(t, x)

    def _calc_ux(self, apply_fn, params, t, x):
        def u_ux(t, x):
            model_fn = lambda xx: self._calc_u(apply_fn, params, t, xx)
            u, du_dx = jax.vjp(model_fn, x)
            ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(u_ux, in_axes=(0, 0))(t, x)

    def _calc_ut_ux(self, apply_fn, params, t, x):
        model_fn = lambda ttxx: self._calc_u(apply_fn, params, ttxx[..., :1], ttxx[..., 1:])
        def u_ut_ux(tx):
            u, du_dx_dt = jax.vjp(model_fn, tx)
            ux_ut = jax.vmap(du_dx_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux_ut[..., :1], ux_ut[..., 1:]
        return jax.vmap(u_ut_ux, in_axes=0)(jnp.concatenate((t, x), axis=-1))

    def _calc_uxx(self, apply_fn, params, t, x):
        def u_ux_uxx(t, x):
            model_fn = lambda xx: self._calc_u(apply_fn, params, t, xx)
            def ux_u(x):
                u, du_dx = jax.vjp(model_fn, x)
                ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            du_dxx = lambda s: jax.jvp(ux_u, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(du_dxx, in_axes=1, out_axes=(None, 1, None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(u_ux_uxx, in_axes=(0, 0))(t, x)

    def _calc_laplacian(self, apply_fn, params, t, x, weight=None):
        weight = jnp.broadcast_to(
            weight if weight is not None else jnp.eye(self.model_config.d_in),
            shape=(x.shape[0], self.model_config.d_in, self.model_config.d_in)
        )
        H = jnp.einsum('bij,bkj->bik', weight, weight)  # H = weight weight^T
        u, _, uxx = self._calc_uxx(apply_fn, params, t, x)
        return u, jnp.einsum('bmij,bij->bm', uxx, H)  # Tr(H uxx)
    
    def _forward_laplacian(self, apply_fn, params, t, x, weight=None):
        return self.model.forward_laplacian(params, t, x, weight=weight)

    # --------------------------------------------------

    def analytic_ut(self, t, x):
        def u_ut(t, x):
            model_fn = lambda tt: self.analytic_u(tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(u_ut, in_axes=(0, 0))(t, x)
    
    def analytic_ux(self, t, x):
        def u_ux(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            u, vjp_fun = jax.vjp(model_fn, x)
            ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(u_ux, in_axes=(0, 0))(t, x)
    
    def analytic_ut_ux(self, t, x):
        model_fn = lambda ttxx: self.analytic_u(ttxx[:1], ttxx[1:])
        def u_ut_ux(tx):
            u, du_dx_dt = jax.vjp(model_fn, tx)
            ux_ut = jax.vmap(du_dx_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux_ut[:1], ux_ut[1:]
        return jax.vmap(u_ut_ux, in_axes=0)(jnp.concatenate((t, x), axis=-1))
    
    def analytic_uxx(self, t, x):
        def u_ux_uxx(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            def ux_x(x):
                u, du_dx = jax.vjp(model_fn, x)
                ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            du_dxx = lambda s: jax.jvp(ux_x, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(du_dxx, in_axes=1, out_axes=(None, 1 ,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(u_ux_uxx, in_axes=(0, 0))(t, x)
    

    # --------------------------------------------------
    # Util Methods for Loss Methods
    # --------------------------------------------------

    def loss_to_grad(self, loss_fn):

        def _loss_and_grad(key, params):
            return jax.value_and_grad(lambda K, P: loss_fn(K, P), argnums=1, has_aux=True)(key, params)

        def grad_fn(key, params):
            n_chunks = (self.solver_config.batch + self.solver_config.micro_batch - 1) // self.solver_config.micro_batch

            def chunk_loop(carry, _):
                key, params, losses_acc, grad_acc = carry
                (total, (losses, key, params)), grad = _loss_and_grad(key, params)
                losses_acc = losses_acc + jnp.asarray(losses)
                grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
                return (key, params, losses_acc, grad_acc), None
            
            losses_0 = jnp.zeros_like(jnp.asarray(loss_fn(key, params)[1][0]))
            grad_0 = jax.tree_util.tree_map(jnp.zeros_like, params)
            (key, params, losses, grad), _ = jax.lax.scan(chunk_loop, (key, params, losses_0, grad_0), None, length=n_chunks)
            losses = losses/n_chunks
            grad = jax.tree_util.tree_map(lambda a: a/n_chunks, grad)
            return key, params, losses, grad
        
        return grad_fn
        

    def make_time_domain(self, key, batch):

        if self.solver_config.use_delta:
            dt = self.solver_config.T / (self.solver_config.traj_len - 1)
            key, sub = jax.random.split(key)
            t1 = jax.random.uniform(sub, (batch, 1), minval=0, maxval=dt)
            diffT = jnp.concatenate([jnp.zeros((batch, 1)),
                                     t1,
                                     jnp.full((batch, self.solver_config.traj_len-2), fill_value=dt), 
                                     jnp.full_like(t1, fill_value=dt) - t1], axis=-1)
            T = jnp.cumsum(diffT, axis=-1)[:, :, None]
        else:
            dt = self.solver_config.T / self.solver_config.traj_len
            T = jnp.broadcast_to(jnp.linspace(0, self.solver_config.T, self.solver_config.traj_len+1)[None, :, None], (batch, self.solver_config.traj_len+1, 1))

        dT = T[:, 1:, :] - T[:, :-1, :]
        return key, T, dT
    

    def make_full_domain_euler(self, key, batch):
        key, T, dT = self.make_time_domain(key, batch)
        
        key, sub = jax.random.split(key)
        dW = jnp.sqrt(dT) * jax.random.normal(sub, (batch, self.solver_config.traj_len, self.model_config.d_in))

        X = jnp.zeros((batch, self.solver_config.traj_len+1, self.model_config.d_in))
        key, X0 = self.get_X0(key, batch)
        X = X.at[:, 0, :].set(X0)

        def loop(i, X):
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :])*dT[:, i-1, :]
                                  + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :]), dW[:, i-1, :, jnp.newaxis])[..., 0])
            return X
        
        X = jax.lax.fori_loop(1, self.solver_config.traj_len+1, loop, X)
        return key, T, dT, dW, X
    
    
    # --------------------------------------------------
    # Loss Methods
    # --------------------------------------------------

    def FSPINNs_loss(self, key, params):
        batch = self.solver_config.micro_batch
        
        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)
        
        pde_loss = self.solver_config.pde_scale * self.pinns_residual(params, T.reshape(-1, 1), X.reshape(-1, self.model_config.d_in))
        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            u, ux = self.calc_ux(params, T[:, -1, :], X[:, -1, :])
            bc_loss = self.model_config.bc_scale * (jnp.mean((u - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((ux - self.calc_bcx(X[:, -1, :]))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
    
    # --------------------------------------------------

    def EMBSDE_loss(self, key, params):
        batch = self.solver_config.micro_batch
        
        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)

        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)

            key, sub = jax.random.split(key)
            dw = jnp.sqrt(dt) * jax.random.normal(sub, (batch, self.model_config.d_in))

            t_new = T[:, i+1, :]
            x_new = x + self.b(t, x)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)

            traj_loss = traj_loss.at[i].set(jnp.mean((u_new - u_calc)**2))
            return key, x_new, u_calc, ux_calc, traj_loss


        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)

    # --------------------------------------------------   

    def HeunBSDE_loss(self, key, params):
        batch = self.solver_config.micro_batch

        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)
        
        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)
            weighted_lap = self.calc_laplacian(params, t, x, weight=sigma)[1]

            key, sub = jax.random.split(key)
            dw = jnp.sqrt(dt) * jax.random.normal(sub, (batch, self.model_config.d_in))
            
            dx_star = self.b_heun(t, x)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_star = x + dx_star
            du_star = (self.h(t, x, u, ux) - self.c(t, x, u, ux, weighted_lap))*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            t_new = T[:, i+1, :]
            u_star, ux_star = self.calc_ux(params, t_new, x_star)
            sigma_star = self.sigma(t_new, x_star)
            weighted_lap_star = self.calc_laplacian(params, t_new, x_star, weight=sigma_star)[1]

            x_new = x + 0.5*dx_star + 0.5*(self.b_heun(t_new, x_star)*dt + jnp.matmul(sigma_star, dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_star + 0.5*((self.h(t_new, x_star, u_star, ux_star) - self.c(t_new, x_star, u_star, ux_star, weighted_lap_star))*dt + jnp.matmul(jnp.matmul(ux_star, sigma_star), dw[..., jnp.newaxis])[..., 0])
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)

            traj_loss = traj_loss.at[i].set(jnp.mean((u_new - u_calc)**2))
            return key, x_new, u_calc, ux_calc, traj_loss


        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
    
    # --------------------------------------------------

    def MultiShotEMBSDE_loss(self, key, params):
        batch = self.solver_config.micro_batch
        main_stack = self.solver_config.main_stack
        d_in = self.model_config.d_in
        d_out = self.model_config.d_out
        
        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)

        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)
            
            t_new = T[:, i+1, :]  # t + dt
            b = self.b(t, x)
            h = self.h(t, x, u, ux)
            Z = jnp.matmul(ux, sigma)
            
            t_stack = jnp.broadcast_to(t_new[:, jnp.newaxis, :], (batch, main_stack, 1))
            x_stack = jnp.broadcast_to((x + b*dt)[:, jnp.newaxis, :], (batch, main_stack, d_in))
            u_stack = jnp.broadcast_to(u[:, jnp.newaxis, :], (batch, main_stack, d_out))
            h_stack = jnp.broadcast_to(h[:, jnp.newaxis, :], (batch, main_stack, d_out))

            key, sub = jax.random.split(key)
            eta_main = jnp.sqrt(dt[:, jnp.newaxis, :]) * jax.random.normal(sub, (batch, main_stack, d_in))

            x_stack_new = x_stack + jnp.einsum('bij,bkj->bki', sigma, eta_main)
            u_stack_new = u_stack + h_stack*dt[:, jnp.newaxis, :] + jnp.einsum('bij,bkj->bki', Z, eta_main)
            u_stack_calc = self.calc_u(params, t_stack, x_stack_new)
            traj_loss = traj_loss.at[i].set(jnp.mean(jnp.mean(u_stack_new - u_stack_calc, axis=-2)**2))

            x_new = x_stack_new[:, 0, :]
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            return key, x_new, u_calc, ux_calc, traj_loss

        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)

    # --------------------------------------------------

    def UnEMBSDE_loss(self, key, params):
        batch = self.solver_config.micro_batch
        main_stack = self.solver_config.main_stack
        sub_stack = self.solver_config.sub_stack
        p = self.solver_config.nobias_Lp
        d_in = self.model_config.d_in
        d_out = self.model_config.d_out
        
        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)

        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)

            t_new = T[:, i+1, :]  # t + dt
            b = self.b(t, x)
            h = self.h(t, x, u, ux)
            Z = jnp.matmul(ux, sigma)
            
            t_stack = jnp.broadcast_to(t_new[:, jnp.newaxis, :], (batch, main_stack + (p-1)*sub_stack, 1))
            x_stack = jnp.broadcast_to((x + b*dt)[:, jnp.newaxis, :], (batch, main_stack + (p-1)*sub_stack, d_in))
            u_stack = jnp.broadcast_to(u[:, jnp.newaxis, :], (batch, main_stack + (p-1)*sub_stack, d_out))
            h_stack = jnp.broadcast_to(h[:, jnp.newaxis, :], (batch, main_stack + (p-1)*sub_stack, d_out))

            key, sub = jax.random.split(key)
            eta_main = jnp.sqrt(dt[:, jnp.newaxis, :]) * jax.random.normal(sub, (batch, main_stack + (p-1)*sub_stack, d_in))

            x_stack_new = x_stack + jnp.einsum('bij,bkj->bki', sigma, eta_main)
            u_stack_new = u_stack + h_stack*dt[:, jnp.newaxis, :] + jnp.einsum('bij,bkj->bki', Z, eta_main)
            u_stack_calc = self.calc_u(params, t_stack, x_stack_new)
            main_loss = (jnp.sum(u_stack_new[:, :main_stack] - u_stack_calc[:, :main_stack], axis=-2))/main_stack
            sub_loss = jnp.prod(jnp.sum(u_stack_new[:, main_stack:].reshape(batch, p-1, sub_stack, 1)
                                        - u_stack_calc[:, main_stack:].reshape(batch, p-1, sub_stack, 1), axis=-2)/sub_stack, axis=-2)
            traj_loss = traj_loss.at[i].set(jnp.mean(main_loss * sub_loss))

            x_new = x_stack_new[:, 0, :]
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            return key, x_new, u_calc, ux_calc, traj_loss

        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
        
    # --------------------------------------------------

    def Shotgun_loss(self, key, params):
        batch = self.solver_config.micro_batch
        d_in = self.model_config.d_in
        d_out = self.model_config.d_out
        Delta_t = self.solver_config.shotgun_Delta_t
        main_stack = self.solver_config.main_stack
        
        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)
        
        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)

            # [1] compute trajectory
            key, sub = jax.random.split(key)
            dw = jnp.sqrt(dt) * jax.random.normal(sub, (batch, d_in))

            t_new = T[:, i+1, :]  # t + dt
            b = self.b(t, x)
            h = self.h(t, x, u, ux)
            Z = jnp.matmul(ux, sigma)

            x_new = x + b*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)

            # [2] compute loss
            t_local = jnp.broadcast_to((t + Delta_t)[:, jnp.newaxis, :], (batch, main_stack, 1))
            x_local = jnp.broadcast_to((x + b*Delta_t)[:, jnp.newaxis, :], (batch, main_stack, d_in))
            u_local = jnp.broadcast_to(u[:, jnp.newaxis, :], (batch, main_stack, d_out))
            h_local = jnp.broadcast_to(h[:, jnp.newaxis, :], (batch, main_stack, d_out))

            key, sub = jax.random.split(key)
            eta = jnp.sqrt(Delta_t) * jax.random.normal(sub, (batch, main_stack, d_in))

            diff = jnp.einsum('bij,bkj->bki', sigma, eta)
            x_local_plus = x_local + diff
            x_local_minus = x_local - diff
            u_local_plus = self.calc_u(params, t_local, x_local_plus)
            u_local_minus = self.calc_u(params, t_local, x_local_minus)

            traj_loss = traj_loss.at[i].set(jnp.mean(jnp.mean((u_local_plus + u_local_minus - 2*u_local)/(2*Delta_t) - h_local, axis=-2)**2))
            return key, x_new, u_calc, ux_calc, traj_loss
        

        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
    
    # --------------------------------------------------
    
    def UnShotgun_loss(self, key, params):
        batch = self.solver_config.micro_batch
        main_stack = self.solver_config.main_stack
        p = self.solver_config.nobias_Lp  # p = 2, 4, ....
        d_in = self.model_config.d_in
        d_out = self.model_config.d_out
        Delta_t = self.solver_config.shotgun_Delta_t
        
        key, T, dT = self.make_time_domain(key, batch)
        key, x_start = self.get_X0(key, batch)
        u_start, ux_start = self.calc_ux(params, T[:, 0, :], x_start)
        traj_loss = jnp.zeros(self.solver_config.traj_len)
        
        def traj_calc(i, inputs):
            key, x, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            sigma = self.sigma(t, x)

            # [1] compute trajectory
            key, sub = jax.random.split(key)
            dw = jnp.sqrt(dt) * jax.random.normal(sub, (batch, d_in))

            t_new = T[:, i+1, :]  # t + dt
            b = self.b(t, x)
            h = self.h(t, x, u, ux)
            Z = jnp.matmul(ux, sigma)

            x_new = x + b*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)

            # [2] compute loss
            t_stack = jnp.broadcast_to((t + Delta_t)[:, jnp.newaxis, jnp.newaxis, :], (batch, p, main_stack, 1))
            x_stack = jnp.broadcast_to((x + b*Delta_t)[:, jnp.newaxis, jnp.newaxis, :], (batch, p, main_stack, d_in))
            u_stack = jnp.broadcast_to(u[:, jnp.newaxis, jnp.newaxis, :], (batch, p, main_stack, d_out))
            h_stack = jnp.broadcast_to(h[:, jnp.newaxis, jnp.newaxis, :], (batch, p, main_stack, d_out))

            key, sub = jax.random.split(key)
            eta = jnp.sqrt(Delta_t) * jax.random.normal(sub, (batch, p, main_stack, d_in))

            diff = jnp.einsum('bij,bpkj->bpki', sigma, eta)
            x_stack_plus = x_stack + diff
            x_stack_minus = x_stack - diff
            u_stack_plus = self.calc_u(params, t_stack, x_stack_plus)
            u_stack_minus = self.calc_u(params, t_stack, x_stack_minus)

            loss = jnp.prod(jnp.mean((u_stack_plus + u_stack_minus - 2*u_stack)/(2*Delta_t) - h_stack, axis=-2), axis=1)
            traj_loss = traj_loss.at[i].set(jnp.mean(loss))
            return key, x_new, u_calc, ux_calc, traj_loss

        key, x_end, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, x_start, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)

        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(x_end))**2) + jnp.mean((ux_end - self.calc_bcx(x_end))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)

    # --------------------------------------------------
    # Plot Methods
    # --------------------------------------------------

    def init_wandb(self):
        print("Initializing wandb")
        wandb.init(project=self.solver_config.project_name,
                   name=self.solver_config.run_name,
                   config={
                       'solver': vars(self.solver_config),
                       'model': vars(self.model_config),
                   })

    def close(self):
        if self.solver_config.save_to_wandb:
            wandb.finish()
        elif hasattr(self, 'log_file'):
            self.log_file.close()

    def get_analytic_sol(self):
        num_traj = 256
        test_dt = self.solver_config.T / self.solver_config.test_traj_len

        T = jnp.broadcast_to(jnp.linspace(0, self.solver_config.T, self.solver_config.test_traj_len+1)[None, :, None], (num_traj, self.solver_config.test_traj_len+1, 1))
        dW = jnp.sqrt(test_dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.model_config.d_in)),                                    
                                                  jax.random.normal(jax.random.key(1), (num_traj, self.solver_config.test_traj_len, self.model_config.d_in))), axis=1)
        W = jnp.cumsum(dW, axis=1)

        X = self.analytic_X(T, W)
        U = jax.lax.scan(lambda _, tx1: (None, jax.lax.scan(lambda _, tx2: (None, self.analytic_u(tx2[0], tx2[1])), None, (tx1[0], tx1[1]))[1]), None, (T, X))[1]
        return T, X, U

    def plot_pred(self, params, i):
        time = self.sol_T[:, :, 0].T
        pred = self.calc_u(params, self.sol_T, self.sol_X)[:, :, 0].T
        true = self.sol_U[:, :, 0].T
        L1 = jnp.mean(jnp.abs(pred - true) / jnp.abs(true), axis=1)
        L2 = jnp.sqrt(jnp.mean(((pred - true) ** 2) / (true ** 2), axis=1))

        fig_pred = plt.figure(figsize=(10, 6))
        plt.plot(time[:, :4], pred[:, :4], "r", linewidth=1)
        plt.plot(time[:, :4], true[:, :4], ":b", linewidth=1)
        plt.title('Prediction') 
        if self.solver_config.save_to_wandb:
            wandb.log({'Prediction': wandb.Image(fig_pred)}, step=i)
        else:
            fig_pred.savefig(self.save_dir / f'Prediction_{i}.png')
        plt.close(fig_pred)

        fig_L1 = plt.figure(figsize=(10, 6))
        plt.plot(time[:, 0], L1, "b", linewidth=1)
        plt.title('L1 Error')
        plt.yscale('log')
        if self.solver_config.save_to_wandb:
            wandb.log({'L1 Error': wandb.Image(fig_L1)}, step=i)
        else:
            fig_L1.savefig(self.save_dir / f'L1_Error_{i}.png')
        plt.close(fig_L1)

        fig_L2 = plt.figure(figsize=(5, 3))
        plt.plot(time[:, 0], L2, "b", linewidth=1)
        plt.title('L2 Error')
        plt.yscale('log')
        if self.solver_config.save_to_wandb:
            wandb.log({'L2 Error': wandb.Image(fig_L2)}, step=i)
        else:
            fig_L2.savefig(self.save_dir / f'L2_Error_{i}.png')
        plt.close(fig_L2)

    def calc_RL(self, params):
        pred = self.calc_u(params, self.sol_T, self.sol_X)
        true = self.sol_U

        RL1 = jnp.mean(jnp.sum(jnp.abs(pred - true), axis=(1, 2)) / jnp.sum(jnp.abs(true), axis=(1, 2)))
        RL2 = jnp.mean(jnp.sqrt(jnp.sum((pred - true)**2, axis=(1, 2)) / jnp.sum(true ** 2, axis=(1, 2))))

        return RL1, RL2
    
    def calc_RL_T0(self, params):
        pred = self.calc_u(params, jnp.zeros(1), self.get_exact_X0())
        true = self.analytic_u(jnp.zeros(1), self.get_exact_X0())

        RL_T0 = jnp.mean(jnp.abs(pred - true) / jnp.abs(true))
        return pred, RL_T0
    
    @partial(jax.jit, static_argnums=0)
    def jit_calc_RL(self,params):
        return self.calc_RL(params)

    @partial(jax.jit, static_argnums=0)
    def jit_calc_RL_T0(self,params):
        return self.calc_RL_T0(params)

    # --------------------------------------------------
    # Optimization Methods
    # --------------------------------------------------
    
    @partial(jax.jit, static_argnums=0)
    def optimize(self, key, params, opt_state):
        key, params, losses, grad = self.grad_fn(key, params)
        loss = jnp.sum(jnp.asarray(losses))
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return key, loss, losses, params, opt_state


# --------------------------------------------------
# Control Class
# --------------------------------------------------

class Controller():

    def __init__(self, solver: Solver, seed=20226074):
        self.solver = solver
        self.key = jax.random.PRNGKey(seed)
        self.key, self.params, self.opt_state = self.solver.init_solver(self.key)

    def step(self, i):
        self.key, loss, losses, self.params, self.opt_state = self.solver.optimize(self.key, self.params, self.opt_state)

        if self.solver.solver_config.save_to_wandb:
            wandb.log({'loss': loss, **{'loss'+str(k+1): v for k, v in dict(enumerate(losses)).items()}}, step=i)
            pred_T0, RL_T0 = self.solver.jit_calc_RL_T0(self.params)
            wandb.log({'pred_T0': pred_T0, "RL_T0": RL_T0}, step=i)

            if self.solver.solver_config.has_traj_sol and (i%(self.solver.solver_config.iter//self.solver.solver_config.num_figures) == 0 or i+1 == self.solver.solver_config.iter):
                RL1, RL2 = self.solver.jit_calc_RL(self.params)
                wandb.log({"RL1": RL1, "RL2": RL2}, step=i)   
                self.solver.plot_pred(self.params, i)
        else:
            log_dict = {'step': i, 'loss': float(loss), **{'loss'+str(k+1): float(v) for k, v in dict(enumerate(losses)).items()}}
            pred_T0, RL_T0 = self.solver.jit_calc_RL_T0(self.params)
            log_dict['pred_T0'] = pred_T0.tolist() if hasattr(pred_T0, 'tolist') else float(pred_T0)
            log_dict['RL_T0'] = float(RL_T0)
            
            if i%(self.solver.solver_config.iter//self.solver.solver_config.num_figures) == 0 or i+1 == self.solver.solver_config.iter:
                log_str = f"[Epoch {i+1:<5}] Loss: {float(loss):.6e} | pred_T0: {float(jnp.mean(pred_T0)):.6e} | RL_T0: {float(RL_T0):.6e}"
                if self.solver.solver_config.has_traj_sol:
                    RL1, RL2 = self.solver.jit_calc_RL(self.params)
                    log_dict['RL1'] = float(RL1)
                    log_dict['RL2'] = float(RL2)
                    log_str += f" | RL1: {float(RL1):.6e} | RL2: {float(RL2):.6e}"
                    self.solver.plot_pred(self.params, i)
                tqdm.tqdm.write(log_str)
                
            self.solver.log_file.write(json.dumps(log_dict) + '\n')
            self.solver.log_file.flush()

    def solve(self):
        for i in tqdm.tqdm(range(self.solver.solver_config.iter)):
            self.step(i)

        path = Path('./checkpoints/')
        path.mkdir(exist_ok=True)
        if self.solver.solver_config.save_model:
            model_bytes = fs.to_bytes(self.params)
            (path/f'{self.solver.solver_config.project_name}_{self.solver.solver_config.run_name}_model.msgpack').write_bytes(model_bytes)
        if self.solver.solver_config.save_opt:
            opt_bytes = fs.to_bytes(self.opt_state)
            (path/f'{self.solver.solver_config.project_name}_{self.solver.solver_config.run_name}_opt.msgpack').write_bytes(opt_bytes)

        self.solver.close()