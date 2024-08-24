from typing import Optional

import click
import safetensors.torch
import torch
import tqdm

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache


@click.command("llm-distance")
@click.argument("model_path", type=str)
@click.argument("secondary_model_path", type=str)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda",
    help="Device to compute on (default: cuda)",
)
def main(
    model_path: str,
    secondary_model_path,
    device: Optional[str],
):
    model = ModelReference.model_validate(model_path)
    secondary_model = ModelReference.model_validate(secondary_model_path)

    cache = LoaderCache()

    for m in tqdm.tqdm([model, secondary_model], desc="Preparing models"):
        cache.get(m)

    model_config = model.config(trust_remote_code=True)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=True)
    )

    loader_1 = cache.get(model)
    loader_2 = cache.get(secondary_model)

    total_distance_squared = 0
    root_mean_square_error = 0
    n = 0

    for weight_info in model_arch_info.all_weights(config=model_config):
        w = loader_1.get_tensor(weight_info.name, device=device)
        w2 = loader_2.get_tensor(weight_info.name, device=device)

        w = w.to(dtype=torch.float32)
        w2 = w2.to(dtype=torch.float32)

        if len(w.shape) == 2 and w.shape != w2.shape:
            _mismatch_index = 0 if w.shape[0] != w2.shape[0] else 1
            _correction = min(w.shape[_mismatch_index], w2.shape[_mismatch_index])
            _a = w[:_correction, :_correction] - w2[:_correction, :_correction]
        else:
            _a =  (w-w2)

        total_distance_squared += torch.sum(_a ** 2).item()
        root_mean_square_error  += torch.sum(_a**2).item()
        n += w.numel()

    total_distance = total_distance_squared ** 0.5
    root_mean_square_error = (root_mean_square_error / n) ** 0.5

    print(f"Total euclidean distance : {total_distance}")
    print(f"Root mean square error: {root_mean_square_error}")


main()
