import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    help="Choice of network architecture")
parser.add_argument("--gpu",
                    type=int,
                    help='Which gpu to use')

args = parser.parse_args()
model = args.model
gpu = args.gpu
depths = [3, 4, 5]

for depth in depths:
    cmd = "python main.py --exec-mode train "
    cmd += "--data examples/brats.json "
    cmd += "--processed-data /workspace/data/mg-nets-results/brats2020/numpy/ "
    cmd += "--results /workspace/data/mg-nets-results/brats2020/results-{}-{}/ ".format(model, depth)
    cmd += "--model {} ".format(model)
    cmd += "--epochs 200 "
    cmd += "--depth {} ".format(depth)
    cmd += "--amp "
    cmd += "--xla "
    cmd += "--seed 42 "
    cmd += "--patch-size 128 128 128 "
    cmd += "--gpus {} ".format(gpu)
    cmd += "--post-no-morph "
    cmd += "--post-no-largest "
    if model == "fmgnet" or model == "wnet":
        cmd += "--pocket "

    subprocess.call(cmd, shell=True)
