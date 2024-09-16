from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlowConfig:
    alpha: float
    cond_label_size: int
    hidden_size: int
    n_blocks: int
    n_hidden: int
    batch_size: int
    with_noise: bool
    use_residual: bool
    activation_fn: str
    dropout_probability: float
    batch_norm: bool
    n_bins: int
    noise_mul: float
    n_epochs: int
    cond_base: bool
    lr: float
    log_interval: int
    dim: int
    particle: str
    base_dir: str
    data_dir_suffix: str
    models_dir_suffix: str
    no_imgs_generated: int
    bnn_ps: bool
    original_ps_scaler: bool
    save_responses: bool
    tail_bound: Optional[int] = field(default=None)
    model_name: Optional[str] = field(default="")
    model_path: Optional[str] = field(default="")
    device: Optional[str] = field(default="")
