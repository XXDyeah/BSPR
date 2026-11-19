# Countering Replay Blindness in Class-Incremental Learning with Blind-Spot Probe Replay

Environment: Linux, Python 3.7, PyTorch 1.10.0 + CUDA 11.3, torchvision 0.11.x, NumPy, SciPy, Pillow, NVIDIA RTX 4090.

```bash
python class_incremental_cosine_cifar100.py --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 --nb_runs 1 --epochs 160 --rs_ratio 0.0 --imprint_weights --lamda 5 --adapt_lamda --dist 0.5 --K 2 --lw_mr 1 --random_seed 1993 --deterministic_training --deterministic_algorithm_policy warn --ckp_prefix baseline
```
```bash
python class_incremental_cosine_cifar100.py --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 --nb_runs 1 --epochs 160 --rs_ratio 0.0 --imprint_weights --lamda 5 --adapt_lamda --dist 0.5 --K 2 --lw_mr 1 --random_seed 1993 --deterministic_training --deterministic_algorithm_policy warn --bspr --ckp_prefix bspr
```
