from pathlib import Path
from autotunenet_bench.metrics.regret import RegretMetric
import json
# import matplotlib.pyplot as plt

output_dir = Path("autotunenet_bench/results/compute_regret")
output_dir.mkdir(parents=True, exist_ok=True)

# baseline = Path("autotunenet_bench/results/cifar_fixed_lr/epoch_metric.json")
# controller = Path("autotunenet_bench/results/cifar_autotunenet/epoch_metric.json")
baseline = Path("autotunenet_bench/results/mnist_fixed_lr/epoch_metric.json")
controller = Path("autotunenet_bench/results/mnist_autotunenet/epoch_metric.json")

regret = RegretMetric(baseline).compute(controller)
# with open(output_dir / "compute_regret_cifar.json", "w") as f:
#     json.dump(regret, f, indent=2)
with open(output_dir / "compute_regret_mnist.json", "w") as f:
    json.dump(regret, f, indent=2)
print(regret)





# plt.plot(regret["per_epoch_regret"])
# plt.axhline(0, color="black", linestyle="--")
# plt.ylabel("Controller − Baseline loss")
# plt.xlabel("Epoch")
# plt.title("Per-epoch regret")
# plt.show()
# python -m autotunenet_bench.metrics.compute_regret