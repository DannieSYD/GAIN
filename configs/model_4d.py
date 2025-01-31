from dataclasses import dataclass, field

from configs.model_3d import SchNet, DimeNet, DimeNetPlusPlus, GemNet, PaiNN, ClofNet, ViSNet, Equiformer, EquiformerV2, GVP, SphereNet, EQGAT, SEGNN


@dataclass
class TransformerPooling:
    num_heads: int = 8
    num_layers: int = 2


@dataclass
class Model4D:
    graph_encoder: str = 'SchNet'
    set_encoder: str = 'Attention'

    schnet: SchNet = field(default_factory=SchNet)
    dimenet: DimeNet = field(default_factory=DimeNet)
    dimenetplusplus: DimeNetPlusPlus = field(default_factory=DimeNetPlusPlus)
    gemnet: GemNet = field(default_factory=GemNet)
    painn: PaiNN = field(default_factory=PaiNN)
    clofnet: ClofNet = field(default_factory=ClofNet)
    visnet: ViSNet = field(default_factory=ViSNet)
    gvp: GVP = field(default_factory=GVP)
    spherenet: SphereNet = field(default_factory=SphereNet)
    eqgat: EQGAT = field(default_factory=EQGAT)
    equiformer: Equiformer = field(default_factory=Equiformer)
    equiformer_v2: EquiformerV2 = field(default_factory=EquiformerV2)
    segnn: SEGNN = field(default_factory=SEGNN)

    transformer: TransformerPooling = field(default_factory=TransformerPooling)
