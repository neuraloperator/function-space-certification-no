import shutil

from lean_no.paths import repo_path
from lean_no.runtime import write_json
from training.burgers_time import (
    config,
    download_data,
    load_data,
    train_model,
)

data_root = repo_path("data/burgers_mini")
model_root = repo_path("models/burgers_1d")
result_root = repo_path("results/burgers_mini")
commit = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
settings = {
    **config,
    "resolution": 16,
    "model": {**config["model"], "n_modes": [16, 16]},
}

def main():
    urls = {}
    for name in ("train", "test"):
        file = f"burgers_{name}_16.pt"
        url = f"https://raw.githubusercontent.com/neuraloperator/neuraloperator/{commit}/neuralop/data/datasets/data/{file}"
        download_data(data_root / file, [url])
        urls[file] = url
    write_json(data_root / "source.json", {"commit": commit, "urls": urls}, strict=True)
    packs, metadata = load_data(data_root, settings)
    working = repo_path(".cache/training/burgers_1d")
    train_model(settings, packs, metadata, working, result_root)
    model_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(working / "best.pt", model_root / "robustness.pt")

if __name__ == "__main__":
    main()
