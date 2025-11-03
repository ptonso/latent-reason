from .labels import DatasetSpec, Factor, EncRule, FactorEncoder, encode_labels_inplace
from typing import List
import numpy as np

def shapes3d_specs() -> DatasetSpec:
    return DatasetSpec([
        Factor("floor_hue",  "cont", lo=0.0, hi=1.0),
        Factor("wall_hue",   "cont", lo=0.0, hi=1.0),
        Factor("object_hue", "cont", lo=0.0, hi=1.0),
        Factor("scale",      "cont", lo=0.75, hi=1.25),
        Factor("shape",      "cat",  n=4),
        Factor("orientation","cont", lo=-30.0, hi=30.0),
    ])

def shapes3d_rules() -> List[EncRule]:
    return [
        EncRule("floor_hue",  cont="to01"),  # [0,1] → [-.5,.5]
        EncRule("wall_hue",   cont="to01"),
        EncRule("object_hue", cont="to01"),
        EncRule("scale",      cont="to01"),
        EncRule("shape",      cat="bin01"),  # {0,1,2,3} → [0..1] bins
        EncRule("orientation",cont="to01"),  # (-365,0] → [0.5,.5]
    ]

if __name__ == "__main__":
    spec, rules = shapes3d_specs(), shapes3d_rules()
    enc = FactorEncoder(spec, rules, default_cont="to01", default_cat="index")

    print("Encoding 3DShapes labels...")
    encode_labels_inplace(
        data_yaml="data/01--clean/3dshapes/data.yaml",
        encoder=enc,
        spec=spec,
        rules=rules,
        splits=("train","val","test"),
    )
