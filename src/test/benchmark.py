import os
import json
import torch
import pandas as pd
from typing import List
from dataclasses import asdict
from torchvision import datasets, transforms

from src.logger import setup_logger
from src.test.config import *
from src.test.trainer import Trainer

def main():

    bm_cfg = BenchmarkConfig()

    datasets_cfg: List[DatasetConfig] = [
        DatasetConfig(
            name="MNIST",
            dataset_cls=datasets.MNIST,
            transform=transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ]),
            input_size=(1, 28, 28),
            num_classes=10
        ),
        DatasetConfig(
            name="CIFAR10",
            dataset_cls=datasets.CIFAR10,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ]),
            input_size=(3, 32, 32),
            num_classes=10
        )
    ]

    models_cfg: List[ModelConfig] = [
        ModelConfig(
            name="CNN",
            model_cls=SimpleCNN,
            args={
                "in_channels": ds.input_size[0],      # type: ignore
                "input_size": (ds.input_size[1],      # type: ignore
                            ds.input_size[2]),     # type: ignore
                "num_classes": ds.num_classes         # type: ignore
            }
        )
        for ds in datasets_cfg
      ] + [
            ModelConfig(
                name="MLP",
                model_cls=SimpleMLP,
                args={
                    "input_dim": ds.input_size[0]
                                * ds.input_size[1]
                                * ds.input_size[2],      # type: ignore
                    "num_classes": ds.num_classes         # type: ignore
                }
            )
            for ds in datasets_cfg
      ]

    bm = Benchmarker(
        model_cfgs=models_cfg,
        dataset_cfgs=datasets_cfg,
        bm_cfg=bm_cfg
    )
    bm.run()



class Benchmarker:
    """Orchestrates benchmarking over multiple models and datasets."""
class Benchmarker:
    """Orchestrates benchmarking over multiple models and datasets."""
    def __init__(
        self,
        model_cfgs: List[ModelConfig],
        dataset_cfgs: List[DatasetConfig],
        bm_cfg: BenchmarkConfig
    ):
        # give your logger its filename
        self.logger      = setup_logger("api.log")
        self.model_cfgs  = model_cfgs
        self.dataset_cfgs= dataset_cfgs

        # extract the fields
        self.seeds       = bm_cfg.seed
        self.batch_size  = bm_cfg.batch_size
        self.epochs      = bm_cfg.epochs
        self.patience    = bm_cfg.patience

        # device might be a string in bm_cfg
        self.device      = (
            torch.device(bm_cfg.device)
            if isinstance(bm_cfg.device, str)
            else bm_cfg.device
        )

    def save_configs(self, output_dir: str) -> None:
        """Save benchmarking configurations to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        cfg = {
            "models": [asdict(m) for m in self.model_cfgs],
            "datasets": [asdict(d) for d in self.dataset_cfgs],
            "seeds": self.seeds,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
        }
        path = os.path.join(output_dir, "configs.json")
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self.logger.info(f"saved configs to {path}")
        except (OSError, IOError) as e:
            self.logger.error(f"failed to save configs: {e}")

    def run(self) -> pd.DataFrame:
        """Run benchmarking and save results."""
        results: List[BenchmarkResult] = []

        for ds_cfg in self.dataset_cfgs:
            for m_cfg in self.model_cfgs:
                # skip mismatches
                if m_cfg.args.get("in_channels", None) is not None \
                   and m_cfg.args["in_channels"] != ds_cfg.input_size[0]:
                    continue

                trainer = Trainer(
                    model_cfg=m_cfg,
                    data_cfg=ds_cfg,
                    device=self.device,
                    seeds=self.seeds
                )
                accs = trainer.train_and_evaluate(
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    patience=self.patience
                )

                mean_acc = sum(accs) / len(accs)
                std_acc  = (sum((a - mean_acc)**2 for a in accs) / len(accs))**0.5

                results.append(
                    BenchmarkResult(
                        method=m_cfg.name,
                        dataset=ds_cfg.name,
                        mean=mean_acc,
                        std=std_acc
                    )
                )

        # build DataFrame & pivot
        df = pd.DataFrame([
            {
                "Method": r.method,
                "Dataset": r.dataset,
                "Result": f"{r.mean:.2f} ± {r.std:.2f}"
            }
            for r in results
        ])
        table = df.pivot(index="Method", columns="Dataset", values="Result")

        # save everything
        out_dir = "benchmarks"
        self.save_configs(out_dir)

        try:
            csv_path = os.path.join(out_dir, "results.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"saved results CSV to {csv_path}")
        except (OSError, IOError) as e:
            self.logger.error(f"failed to save CSV: {e}")

        try:
            md_path = os.path.join(out_dir, "results.md")
            with open(md_path, "w") as f:
                f.write(table.to_markdown())
            self.logger.info(f"saved markdown table to {md_path}")
        except (OSError, IOError) as e:
            self.logger.error(f"failed to save markdown: {e}")

        # log final table
        self.logger.info("\n" + table.to_markdown())
        return table


if __name__ == "__main__":
    main()