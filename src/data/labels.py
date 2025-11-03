from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict
import numpy as np
from pathlib import Path
import yaml
from src.vae.hydra_io import save_yaml, load_config

ContMode = Literal["to01", "center01"]
CatMode  = Literal["bin01", "index", "onehot"]

@dataclass(frozen=True)
class Factor:
    name: str
    kind: Literal["cont", "cat"]
    lo:   Optional[float] = None
    hi:   Optional[float] = None
    n:    Optional[int]   = None

@dataclass(frozen=True)
class DatasetSpec:
    factors: List[Factor]
    def by_name(self) -> Dict[str, Factor]:
        return {f.name: f for f in self.factors}
    def dim_raw(self) -> int:
        return len(self.factors)

@dataclass(frozen=True)
class EncRule:
    name: str
    cont: Optional[ContMode] = None
    cat:  Optional[CatMode]  = None

class FactorEncoder:
    def __init__(
        self,
        spec: DatasetSpec,
        rules: List[EncRule],
        default_cont: ContMode = "to01",
        default_cat:  CatMode  = "bin01",
    ):
        self.spec = spec
        self._rules = {r.name: r for r in rules}
        self.default_cont = default_cont
        self.default_cat  = default_cat
        self._slices: List[Tuple[int, int]] = []
        self._out_dim = 0
        for f in self.spec.factors:
            mode = self._cat_mode(f.name) if f.kind == "cat" else None
            if f.kind == "cont":
                self._slices.append((self._out_dim, self._out_dim + 1)); self._out_dim += 1
            elif mode == "onehot":
                n = int(f.n or 0); 
                if n <= 0: raise ValueError(f"{f.name}: n required")
                self._slices.append((self._out_dim, self._out_dim + n)); self._out_dim += n
            else:
                self._slices.append((self._out_dim, self._out_dim + 1)); self._out_dim += 1

    def out_dim(self) -> int:
        return self._out_dim

    def _cat_mode(self, fname: str) -> CatMode:
        r = self._rules.get(fname)
        return r.cat if (r and r.cat) else self.default_cat

    def _cont_mode(self, fname: str) -> ContMode:
        r = self._rules.get(fname)
        return r.cont if (r and r.cont) else self.default_cont

    @staticmethod
    def _bin_center(k: int, n: int) -> float:
        return (k + 0.5) / float(n)

    @staticmethod
    def _nearest_bin(y: float, n: int) -> int:
        centers = (np.arange(n, dtype=np.float32) + 0.5) / float(n)
        return int(np.clip(np.argmin(np.abs(centers - y)), 0, n - 1))

    def encode(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, np.float32).reshape(-1)
        if x.size != self.spec.dim_raw():
            raise ValueError("raw dim mismatch")
        out = np.full((self._out_dim,), np.nan, np.float32)
        for i, f in enumerate(self.spec.factors):
            s0, s1 = self._slices[i]
            if f.kind == "cont":
                v = x[i]
                if np.isnan(v): out[s0] = np.nan; continue
                if f.lo is None or f.hi is None: raise ValueError(f"{f.name}: lo/hi required")
                y = (v - f.lo) / max(1e-8, (f.hi - f.lo))
                if self._cont_mode(f.name) == "center01": y = y - 0.5
                out[s0] = np.float32(y)
            else:
                n = int(f.n or 0)
                if n <= 0: raise ValueError(f"{f.name}: n required")
                v = x[i]
                if np.isnan(v):
                    out[s0:s1] = np.nan; continue
                k = int(np.clip(round(v), 0, n - 1))
                mode = self._cat_mode(f.name)
                if mode == "index":
                    out[s0] = np.float32(k)
                elif mode == "bin01":
                    out[s0] = np.float32(self._bin_center(k, n))
                else:
                    yy = np.zeros((n,), np.float32); yy[k] = 1.0
                    out[s0:s1] = yy
        return out

    def decode(self, enc: np.ndarray) -> np.ndarray:
        z = np.asarray(enc, np.float32).reshape(-1)
        if z.size != self._out_dim:
            raise ValueError("enc dim mismatch")
        raw = np.full((self.spec.dim_raw(),), np.nan, np.float32)
        for i, f in enumerate(self.spec.factors):
            s0, s1 = self._slices[i]
            if f.kind == "cont":
                y = z[s0]
                if np.isnan(y): raw[i] = np.nan; continue
                if f.lo is None or f.hi is None: raise ValueError(f"{f.name}: lo/hi required")
                if self._cont_mode(f.name) == "center01": y = y + 0.5
                v = y * (f.hi - f.lo) + f.lo
                raw[i] = np.float32(v)
            else:
                n = int(f.n or 0)
                seg = z[s0:s1]
                if np.isnan(seg).all(): raw[i] = np.nan; continue
                mode = self._cat_mode(f.name)
                if mode == "index":
                    raw[i] = float(int(np.clip(round(seg[0]), 0, n - 1)))
                elif mode == "bin01":
                    raw[i] = float(self._nearest_bin(float(seg[0]), n))
                else:
                    k = int(np.nanargmax(seg[:n] if seg.size >= n else seg))
                    raw[i] = float(np.clip(k, 0, n - 1))
        return raw

def encode_labels_inplace(data_yaml: str, encoder, spec, rules, splits=("train","val","test"), fmt="%.6f"):
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    for s in splits:
        d = Path(cfg[f"{s}_labels"])
        for p in sorted(d.glob("*.txt")):
            a = np.loadtxt(p, ndmin=1, dtype=np.float32).reshape(-1)
            z = encoder.encode(a).astype(np.float32)
            np.savetxt(p, z[None, :], fmt=fmt)
    cfg_dir = Path(data_yaml).parent
    save_yaml(cfg_dir / "dataset_spec.yaml", spec)
    save_yaml(cfg_dir / "encoder_rules.yaml", rules)

def load_spec_and_rules(spec_yaml: str, rules_yaml: str):
    spec  = load_config(Path(spec_yaml),  DatasetSpec)
    rules = load_config(Path(rules_yaml), List[EncRule])
    return spec, rules


if __name__ == "__main__":
    from .encode_3dshapes import shapes3d_specs, shapes3d_rules
    spec, rules = shapes3d_specs(), shapes3d_rules()
    enc = FactorEncoder(spec, rules, default_cont="to01", default_cat="index")

    np.set_printoptions(
        precision=2,     # digits after decimal
        floatmode="fixed"
    )
    raw = np.array([0.2, 0.7, 0.1, 0.5, 1.0, -73.0], np.float32)
    z   = enc.encode(raw)      # shape: 4 cont + 4 one-hot + 1 cont = 9 dims
    raw_back = enc.decode(z)   # round-trip to original units
    print("raw:     ", raw)
    print("encoded: ", z)
    print("decoded: ", raw_back)