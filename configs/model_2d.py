from dataclasses import dataclass, field


@dataclass
class GIN:
    num_layers: int = 6
    virtual_node: bool = False


@dataclass
class GPS:
    num_layers: int = 6
    walk_length: int = 20
    num_heads: int = 4
    layer_type: str = 'GINE + Transformer'


@dataclass
class ChemProp:
    num_layers: int = 6


@dataclass
class Model2D:
    model: str = 'GPS'
    gin: GIN = field(default_factory=GIN)
    gps: GPS = field(default_factory=GPS)
    chemprop: ChemProp = field(default_factory=ChemProp)
