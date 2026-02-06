import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from learning_rules_MLP import MLP, init_signed_lognormal_weights


def main() -> None:
    torch.manual_seed(0)

    model = MLP([1, 32, 128, 256, 512, 1], activation=torch.sigmoid, output_activation=None)
    init_signed_lognormal_weights(
        model,
        log_mu=-2.0,
        log_sigma=1.0,
        p_inhib=0.2,
        by_neuron=False,
    )

    model.plot_weight_distributions(
        bins=60,
        include_bias=True,
        title="Signed log-normal weight init",
        show=True,
    )


if __name__ == "__main__":
    main()
