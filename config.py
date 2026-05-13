from dataclasses import dataclass, field

@dataclass
class Model_Config():
    use_float64: bool = False
    
    # dimensions
    d_in: int = 1  # not including t
    d_out: int = 1

    # method of computing laplacian
    laplacian_method: str = 'backward'  # 'forward' allowed when the model has 'forward_laplacian' function

    # problem setting in model
    T: float = 1.0
    bc_name: str = 'default'  # We can define additional boundary conditions in 'models.py'
    use_hard_constraint: bool = False
    bc_scale: float = 1.0  # if use_hard_constraint = False

    # model 
    model_name: str = 'MLP'  # We can define additional models in 'models.py'

    # MLP model setting
    MLP_d_hidden: int = 512
    MLP_num_layers: int = 4
    MLP_activation: int = 'mish'
    MLP_skip_conn: bool = False
    MLP_save_layers: tuple = (0,2)
    MLP_skip_layers: tuple = (2,4)
    MLP_kernel_init: str = 'xavier_init'


@dataclass
class Solver_Config():

    # iteration
    iter: int = 10000

    # batch
    batch: int = 64
    micro_batch: int = 64

    # optimizer and scheduler
    optim: str = 'adam'  # 'adam', 'adamw'
    lr: float = 1e-3
    schedule: str = 'cosine_decay'  # 'piecewise_constant', 'cosine_decay', 'cosine_onecycle'
    boundaries_and_scales: dict = field(default_factory=lambda: {2500: 0.1, 5000: 0.1, 7500: 0.1})  # for piecewise_constant
    
    # problem setting in solver
    problem_name: str = 'HJB'
    T: float = 1.0

    # trajectory setting
    X0_std: float = 0.0  # X0 is sampled from N(0, X0_std)
    traj_len: int = 100
    use_delta: bool = False  # if use_delta is True, t_1 is sampled from U(0, T/traj_len)

    # loss method
    loss_method: str = 'EMBSDE'  # 'EMBSDE', 'Shotgun', 'MultiShotEMBSDE', 'HeunBSDE', 'FSPINNs', 'UnEMBSDE', 'UnShotgun'
    pde_scale: float = 1.0

    # stack setting
    main_stack: int = 1
    sub_stack: int = 1  # for UnEMBSDE
    shotgun_Delta_t: float = 4**(-5)

    # nobias loss setting
    nobias_Lp: int = 2


    # checkpointing
    checkpointing: bool = True

    # evaluation
    test_traj_len: int = 100
    has_traj_sol: bool = True

    # save and load
    save_model: bool = True
    save_opt: bool = False
    model_state: str = 'address'
    opt_state: str = 'address'

    # project
    project_name: str = 'UnEMBSDE_experiments'
    run_name: str = 'test'
    save_to_wandb: bool = True
    num_figures: int = 100