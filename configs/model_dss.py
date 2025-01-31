from dataclasses import dataclass, field

from configs.model_3d import SchNet, DimeNet, DimeNetPlusPlus, GemNet, PaiNN, ClofNet, Equiformer, EquiformerV2, ViSNet, GVP, SphereNet, EQGAT, SEGNN



@dataclass
class ModelDSS:
    conf_encoder: str = 'Equiformer'
    topo_encoder: str = 'GPS'

    schnet: SchNet = field(default_factory=SchNet)
    dimenet: DimeNet = field(default_factory=DimeNet)
    dimenetplusplus: DimeNetPlusPlus = field(default_factory=DimeNetPlusPlus)
    gemnet: GemNet = field(default_factory=GemNet)
    painn: PaiNN = field(default_factory=PaiNN)
    clofnet: ClofNet = field(default_factory=ClofNet)
    equiformer: Equiformer = field(default_factory=Equiformer)
    equiformer_v2: EquiformerV2 = field(default_factory=EquiformerV2)
    visnet: ViSNet = field(default_factory=ViSNet)
    gvp: GVP = field(default_factory=GVP)
    spherenet: SphereNet = field(default_factory=SphereNet)
    eqgat: EQGAT = field(default_factory=EQGAT)
    segnn: SEGNN = field(default_factory=SEGNN)

