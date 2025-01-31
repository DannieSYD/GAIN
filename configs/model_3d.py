from dataclasses import dataclass, field
from typing import Optional
from torch import Tensor


@dataclass
class SchNet:
    hidden_dim: int = 128
    num_filters: int = 5
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: int = 10
    readout: str = 'mean'
    dipole: bool = False


@dataclass
class DimeNet:
    hidden_channels: int = 128
    out_channels: int = 128
    num_blocks: int = 6
    num_bilinear: int = 8
    num_spherical: int = 7
    num_radial: int = 6


@dataclass
class DimeNetPlusPlus:
    hidden_channels: int = 128
    out_channels: int = 128
    num_blocks: int = 4
    int_emb_size: int = 64
    basis_emb_size: int = 8
    out_emb_channels: int = 256
    num_spherical: int = 7
    num_radial: int = 6


@dataclass
class GemNet:
    num_spherical: int = 7
    num_radial: int = 6
    num_blocks: int = 4
    emb_size_atom: int = 128
    emb_size_edge: int = 128
    emb_size_trip: int = 64
    emb_size_rbf: int = 16
    emb_size_cbf: int = 16
    emb_size_bil_trip: int = 64
    num_before_skip: int = 1
    num_after_skip: int = 1
    num_concat: int = 1
    num_atoms: int = 1
    num_atom: int = 2
    bond_feat_dim: int = 0  # unused_argument


@dataclass
class ChIRo:
    F_z_list: list = field(default_factory=lambda: [8, 8, 8])
    F_H: int = 64
    F_H_EConv: int = 64
    layers_dict: dict = field(default_factory=lambda: {
        'EConv_mlp_hidden_sizes': [32, 32],
        'GAT_hidden_node_sizes': [64],
        'encoder_hidden_sizes_D': [64, 64],
        'encoder_hidden_sizes_phi': [64, 64],
        'encoder_hidden_sizes_c': [64, 64],
        'encoder_hidden_sizes_alpha': [64, 64],
        'encoder_hidden_sizes_sinusoidal_shift': [256, 256],
        'output_mlp_hidden_sizes': [128, 128]
    })
    GAT_N_heads: int = 4
    chiral_message_passing: bool = False
    CMP_EConv_MLP_hidden_sizes: list = field(default_factory=lambda: [256, 256])
    CMP_GAT_N_layers: int = 3
    CMP_GAT_N_heads: int = 2
    c_coefficient_normalization: str = 'sigmoid'
    encoder_reduction: str = 'sum'
    output_concatenation_mode: str = 'both'


@dataclass
class ClofNet:
    cutoff: int = 6.5
    num_layers: int = 6
    hidden_channels: int = 128
    num_radial: int = 32


@dataclass
class PaiNN:
    hidden_dim: int = 128
    num_interactions: int = 3  # 3
    num_rbf: int = 64  # 20
    cutoff: float = 12.0  # 5.0
    readout: str = 'add'  # 'add' or 'mean'
    # activation: Optional[Callable] = F.silu
    shared_interactions: bool = False
    shared_filters: bool = False


@dataclass
class ViSNet:
    lmax: int = 1
    trainable_vecnorm: bool = False
    num_heads: int = 8
    num_layers: int = 6
    hidden_channels: int = 128
    num_rbf: int = 32
    trainable_rbf: bool = False
    cutoff: float = 5.0
    max_num_neighbors: int = 32
    vertex: bool = False
    reduce_op: str = "sum"
    mean: float = 0.0
    std: float = 1.0
    derivative: bool = False


@dataclass
class GVP:
    hidden_dim: int = 128
    num_rbf: int = 32
    num_layers: int = 6
    edge_cutoff: float = 5.0

@dataclass
class SphereNet:
    cutoff: float = 5.0
    num_layers: int = 4
    hidden_channels: int = 128
    out_channels: int = 128


@dataclass
class EQGAT:
    hidden_dim: int = 128
    num_layers: int = 5
    eps: float = 1e-6
    cutoff: float = 5.0
    num_radial: int = 32
    use_norm: bool = False
    basis: str = "bessel"
    use_mlp_update: bool = True
    use_cross_product: bool = True
    no_feat_attn: bool = False
    vector_aggr: str = "mean"
    max_num_neighbors: int = 32
    num_rbf: int = 32
    trainable_rbf: bool = False
    max_z: int = 100

@dataclass
class SEGNN:
    input_features: int = 128*2
    # input_features: int = 128
    output_features: int = 128
    hidden_features: int = 128*2
    # hidden_features: int = 128
    N: int = 6
    norm: str = "instance"
    lmax_h: int = 1
    lmax_pos: float = None
    pool: str = "avg"
    edge_inference: bool = False
    cutoff: float = 5.0


@dataclass
class Equiformer:
    # irreps_node_embedding: str = '128x0e+64x1e+32x2e'
    irreps_node_embedding: str = '128x0e+128x1e'
    # num_layers: int = 6
    num_layers: int = 2
    irreps_sh: str = '1x0e+1x1e+1x2e'
    max_radius: float = 5.0
    number_of_basis: int = 128
    fc_neurons: list = field(default_factory=lambda: [64, 64])
    irreps_feature: str = '128x0e'
    # irreps_feature: str = '128x0e+128x1e'
    irreps_head: str = '32x0e+16x1e+8x2e'
    num_heads: int = 4
    irreps_pre_attn: str = None
    rescale_degree: bool = False
    nonlinear_message: bool = True
    irreps_mlp_mid: str = '384x0e+192x1e+96x2e'
    norm_layer: str = 'layer'
    alpha_drop: float = 0.2
    proj_drop: float = 0.0
    out_drop: float = 0.0
    drop_path_rate: float = 0.0
    scale: float = None


@dataclass
class EquiformerV2:
    use_pbc: bool = False
    regress_forces: bool = False
    otf_graph: bool = True
    max_neighbors: int = 500
    max_radius: float = 5.0
    max_num_elements: int = 90

    num_layers: int = 6
    sphere_channels: int = 96
    attn_hidden_channels: int = 48
    num_heads: int = 4
    attn_alpha_channels: int = 64
    attn_value_channels: int = 24
    ffn_hidden_channels: int = 96

    norm_type: str = 'rms_norm_sh'

    lmax_list: list = field(default_factory=lambda: [4])
    mmax_list: list = field(default_factory=lambda: [4])
    grid_resolution: int = None

    num_sphere_samples: int = 128

    edge_channels: int = 128
    use_atom_edge_embedding: bool = True
    share_atom_edge_embedding: bool = False
    use_m_share_rad: bool = False
    distance_function: str = "gaussian"
    num_distance_basis: int = 128

    attn_activation: str = 'scaled_silu'
    use_s2_act_attn: bool = False
    use_attn_renorm: bool = True
    ffn_activation: str = 'scaled_silu'
    use_gate_act: bool = False
    use_grid_mlp: bool = False
    use_sep_s2_act: bool = True

    alpha_drop: float = 0.1
    drop_path_rate: float = 0
    proj_drop: float = 0.0

    weight_init: str = 'normal'


@dataclass
class Model3D:
    model: str = 'PaiNN'
    augmentation: bool = True

    schnet: SchNet = field(default_factory=SchNet)
    dimenet: DimeNet = field(default_factory=DimeNet)
    dimenetplusplus: DimeNetPlusPlus = field(default_factory=DimeNetPlusPlus)
    gemnet: GemNet = field(default_factory=GemNet)
    chiro: ChIRo = field(default_factory=ChIRo)
    painn: PaiNN = field(default_factory=PaiNN)
    clofnet: ClofNet = field(default_factory=ClofNet)
    visnet: ViSNet = field(default_factory=ViSNet)
    equiformer: Equiformer = field(default_factory=Equiformer)
    equiformer_v2: EquiformerV2 = field(default_factory=EquiformerV2)
