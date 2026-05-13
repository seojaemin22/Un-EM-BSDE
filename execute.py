from solver import *
from config import *
from problems import *
import jax
from datetime import datetime
import argparse

parser = argparse.ArgumentParser("Solve High-Dimensional PDE")

parser.add_argument("--problem", type=str, choices=['HJB', 'BSB', 'AC'])
parser.add_argument("--loss", type=str, default='UnEMBSDE', choices=['EMBSDE', 'Shotgun', 'MultiShotEMBSDE', 'HeunBSDE', 'FSPINNs', 'UnEMBSDE', 'UnShotgun'])
parser.add_argument("--hardcon", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100000)

parser.add_argument("--d_in", type=int, default=100)
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument("--traj_len", type=int, default=100)

parser.add_argument("--main_stack", type=int, default=5)
parser.add_argument("--sub_stack", type=int, default=5)

parser.add_argument("--float64", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--num_figures', type=int, default=100)
parser.add_argument('--project_name', type=str, default='UnEMBSDE_experiments')
parser.add_argument('--run_name', type=str, default=None)

args = parser.parse_args()
gparams = args.__dict__


use_float64 = bool(args.float64)
if use_float64:
    jax.config.update('jax_enable_x64', True)

model_config = Model_Config()
solver_config = Solver_Config()

model_config.d_in = gparams['d_in']
model_config.T = gparams['T']
model_config.use_hard_constraint = bool(args.hardcon)

solver_config.iter = gparams['epochs']
solver_config.batch = gparams['batch']
solver_config.problem_name = gparams['problem']
solver_config.T = gparams['T']
solver_config.traj_len = gparams['traj_len']
solver_config.loss_method = gparams['loss']
solver_config.main_stack = gparams['main_stack']
solver_config.sub_stack = gparams['sub_stack']
solver_config.use_delta = True if gparams['loss'] in ['Shotgun', 'UnShotgun'] else False

solver_config.save_to_wandb = bool(args.wandb)
solver_config.project_name = gparams['project_name']
solver_config.run_name = f"{gparams['problem']}_{gparams['loss']}_hardcon{bool(args.hardcon)}_{gparams['seed']}" if gparams['run_name'] is None else gparams['run_name']
solver_config.num_figures = gparams['num_figures']

if solver_config.problem_name == 'AC':
    model_config.d_in = 20
    model_config.T = 0.3
    solver_config.T = 0.3
    solver_config.has_traj_sol = False

if solver_config.loss_method in ['FSPINNs', 'HeunBSDE']:
    solver_config.micro_batch = 16

svr = SOLVER_REGISTRY[solver_config.problem_name](model_config, solver_config)
ctr = Controller(svr, seed=gparams['seed'])
ctr.solve()

if use_float64:
    jax.config.update("jax_enable_x64", False)