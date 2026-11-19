#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import numpy as np
import os
import sys
import copy
import argparse
import math
import json
import random
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bspr.checkpoint import (
    companion_filename,
    load_bspr_checkpoint,
    save_bspr_checkpoint,
    validate_bspr_checkpoint_identity,
)
from bspr.config import add_bspr_arguments, resolve_bspr_config
from bspr.controller import evaluate_reference_risks
from bspr.state import BSPRState


_SEED_MASK_64 = (1 << 64) - 1
_SEED_MASK_63 = (1 << 63) - 1


def resolve_training_seed(random_seed, training_seed, deterministic_training):
    if not deterministic_training:
        return None
    if training_seed is None:
        training_seed = random_seed
    return int(training_seed) & _SEED_MASK_63


def derive_stage_seed(training_seed, run_index, stage_index, stream_index=0):
    if training_seed is None:
        raise ValueError('training_seed must be resolved before deriving stage seeds')
    value = int(training_seed) & _SEED_MASK_64
    value ^= ((int(run_index) + 1) * 0x9E3779B97F4A7C15) & _SEED_MASK_64
    value ^= ((int(stage_index) + 1) * 0xBF58476D1CE4E5B9) & _SEED_MASK_64
    value ^= ((int(stream_index) + 1) * 0x94D049BB133111EB) & _SEED_MASK_64
    value = (value + 0x9E3779B97F4A7C15) & _SEED_MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _SEED_MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _SEED_MASK_64
    value ^= value >> 31
    return value & _SEED_MASK_63


def seed_runtime_random_generators(seed):
    seed = int(seed) & _SEED_MASK_63
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_training(
        enabled, training_seed, algorithm_policy='warn'):
    if not enabled:
        return 'disabled'
    if training_seed is None:
        raise ValueError('deterministic training requires an effective training seed')
    if algorithm_policy not in ('warn', 'strict'):
        raise ValueError("algorithm_policy must be 'warn' or 'strict'")
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if algorithm_policy == 'strict':
        torch.use_deterministic_algorithms(True)
        effective_policy = 'strict'
    else:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            effective_policy = 'warn_only'
        except TypeError:
            if hasattr(torch, 'set_deterministic_debug_mode'):
                try:
                    torch.set_deterministic_debug_mode('warn')
                except (TypeError, ValueError):
                    torch.set_deterministic_debug_mode(1)
                effective_policy = 'warn_only_debug_api'
            else:
                torch.use_deterministic_algorithms(False)
                effective_policy = 'cudnn_seeded'
    seed_runtime_random_generators(training_seed)
    return effective_policy


def seed_data_loader_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed) & _SEED_MASK_63)
    return generator


def make_seeded_data_loader(dataset, deterministic_seed=None, **kwargs):
    if deterministic_seed is None:
        return torch.utils.data.DataLoader(dataset, **kwargs)
    kwargs['worker_init_fn'] = seed_data_loader_worker
    kwargs['generator'] = make_torch_generator(deterministic_seed)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def make_weighted_random_sampler(weights, num_samples, deterministic_seed=None):
    generator = (
        None if deterministic_seed is None
        else make_torch_generator(deterministic_seed)
    )
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples, generator=generator)


class TeeStream(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return self.streams[0].isatty()

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')

def load_base50_model(model, checkpoint_path, device, expected_num_classes):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        loaded_model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get('model'), nn.Module):
        loaded_model = checkpoint['model']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            'model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict)
        loaded_model = model
    else:
        raise TypeError(
            'Unsupported Base50 checkpoint type: {}'.format(type(checkpoint).__name__))

    actual_num_classes = getattr(loaded_model.fc, 'out_features', None)
    if actual_num_classes != expected_num_classes:
        raise ValueError(
            'Base50 checkpoint has {} output classes, expected {}.'.format(
                actual_num_classes, expected_num_classes))
    return loaded_model.to(device)


def get_incremental_training_file(bspr_enabled, incremental_stage):
    filename = (
        'incremental_train_and_eval_MR_LF_BSPR.py'
        if bspr_enabled and incremental_stage
        else 'incremental_train_and_eval_MR_LF.py'
    )
    return os.path.join('utils_incremental', filename).replace('\\', '/')

import modified_resnet_cifar
import modified_linear
import utils_pytorch
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import (
    compute_accuracy, print_cumulative_accuracy_summary)
from utils_incremental.incremental_train_and_eval_MR_LF import incremental_train_and_eval_MR_LF

NUM_CLASSES = 100
DATASET_NAME = 'cifar100'

parser = argparse.ArgumentParser(
    description='CIFAR-100 class-incremental LUCIR with optional BSPR')
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
    help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
    help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, \
    help='Epochs')
parser.add_argument('--resume', action='store_true', \
    help='resume from the latest matching experiment checkpoint')
parser.add_argument('--use_base50_checkpoint', action='store_true', \
    help='load checkpoint/Base50_Trained_model.pth and skip base-50 training')
parser.add_argument('--rs_ratio', default=0, type=float, \
    help='The ratio for resample')
parser.add_argument('--imprint_weights', action='store_true', \
    help='Imprint the weights for novel classes')
parser.add_argument('--lamda', default=5, type=float, \
    help='Lamda for LF')
parser.add_argument('--adapt_lamda', action='store_true', \
    help='Adaptively change lamda')
parser.add_argument('--dist', default=0.5, type=float, \
    help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=2, type=int, \
    help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, \
    help='loss weight for margin ranking loss')
parser.add_argument('--random_seed', default=1993, type=int, \
    help='class-order seed')
parser.add_argument('--training_seed', default=None, type=int, \
    help='training RNG seed; used only with --deterministic_training and '
         'defaults to --random_seed when omitted')
parser.add_argument('--deterministic_training', action='store_true', \
    help='make model initialization, data order, augmentation, and Torch '
         'execution reproducible')
parser.add_argument('--deterministic_algorithm_policy', default='warn', \
    choices=('warn', 'strict'), \
    help='behavior when this PyTorch lacks a deterministic CUDA kernel: '
         'warn and continue (default) or fail fast')
parser.add_argument('--allow_cpu_training', action='store_true', \
    help='explicitly permit CPU training; by default CUDA unavailability is '
         'a fatal startup error to prevent an accidental silent fallback')
add_bspr_arguments(parser)
args = parser.parse_args()
if (args.nb_cl <= 0 or args.nb_cl_fg < args.nb_cl
        or args.nb_cl_fg % args.nb_cl != 0
        or NUM_CLASSES % args.nb_cl != 0):
    raise ValueError('class counts must satisfy nb_cl > 0, nb_cl_fg >= nb_cl, '
                     'and both nb_cl_fg and 100 must be divisible by nb_cl')
if args.nb_protos <= 0 or args.epochs <= 0 or args.nb_runs <= 0:
    raise ValueError('nb_protos, epochs, and nb_runs must be positive')
if args.K <= 0 or args.K > args.nb_cl:
    raise ValueError('K must be in [1, nb_cl]')
if args.lamda < 0 or args.dist < 0 or args.lw_mr < 0:
    raise ValueError('lamda, dist, and lw_mr must be non-negative')
if not 0.0 <= args.rs_ratio < 1.0:
    raise ValueError('rs_ratio must be in [0, 1)')
effective_training_seed = resolve_training_seed(
    args.random_seed, args.training_seed, args.deterministic_training)
deterministic_algorithm_policy_effective = 'disabled'
if args.deterministic_training:
    deterministic_algorithm_policy_effective = configure_deterministic_training(
        True,
        effective_training_seed,
        args.deterministic_algorithm_policy,
    )
    class_order_rng = np.random.RandomState(args.random_seed)
else:
    np.random.seed(args.random_seed)
    class_order_rng = None
bspr_config = resolve_bspr_config(args)

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
cuda_probe_error = None
cuda_available = bool(torch.cuda.is_available())
try:
    cuda_device_count = int(torch.cuda.device_count())
except Exception as error:
    cuda_device_count = 0
    cuda_available = False
    cuda_probe_error = '{}: {}'.format(type(error).__name__, str(error))
cuda_device_name = None
cuda_device_names = []
cuda_context_probe_ok = False
if cuda_available and cuda_device_count > 0:
    try:
        cuda_device_names = [
            str(torch.cuda.get_device_name(index))
            for index in range(cuda_device_count)
        ]
        cuda_device_name = cuda_device_names[0]
        torch.cuda.set_device(0)
        cuda_probe_tensor = torch.empty(1, device='cuda:0')
        cuda_probe_tensor.add_(1)
        torch.cuda.synchronize()
        del cuda_probe_tensor
        cuda_context_probe_ok = True
    except Exception as error:
        cuda_available = False
        cuda_probe_error = '{}: {}'.format(type(error).__name__, str(error))
if cuda_available and cuda_device_count > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
runtime_device_audit = {
    'cuda_visible_devices': cuda_visible_devices,
    'torch_version': torch.__version__,
    'torch_cuda_build': torch.version.cuda,
    'cuda_available': bool(cuda_available),
    'cuda_device_count': int(cuda_device_count),
    'cuda_device_name': cuda_device_name,
    'cuda_device_names': cuda_device_names,
    'cuda_context_probe_ok': bool(cuda_context_probe_ok),
    'cuda_probe_error': cuda_probe_error,
    'selected_device': str(device),
    'allow_cpu_training': bool(args.allow_cpu_training),
}
print('[Runtime device audit] {}'.format(
    json.dumps(runtime_device_audit, sort_keys=True)), flush=True)
if device.type != 'cuda' and not args.allow_cpu_training:
    raise RuntimeError(
        'CUDA is unavailable; refusing silent CPU fallback. Runtime audit: {}. '
        'Verify that the runtime exposes the intended accelerator. Add '
        '--allow_cpu_training only for an intentional CPU run.'
        .format(json.dumps(runtime_device_audit, sort_keys=True)))

train_batch_size = 128
test_batch_size = 100
eval_batch_size = 128
base_lr = 0.1
lr_strat = [80, 120]
lr_factor = 0.1
custom_weight_decay = 5e-4
custom_momentum = 0.9
args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)

run_timestamp = datetime.now().strftime('%Y-%m%d-%H%M')
checkpoint_root = os.path.join(SCRIPT_DIR, 'checkpoint')
os.makedirs(checkpoint_root, exist_ok=True)
experiment_config_name = args.ckp_prefix.replace('/', '_').replace('\\', '_')
experiment_dir_base = os.path.join(
    checkpoint_root, '{}_v{}'.format(experiment_config_name, run_timestamp))

experiment_dir = experiment_dir_base
duplicate_index = 2
while os.path.exists(experiment_dir):
    experiment_dir = '{}_{:02d}'.format(experiment_dir_base, duplicate_index)
    duplicate_index += 1
os.makedirs(experiment_dir)

if bspr_config.enabled:
    with open(os.path.join(
            experiment_dir, 'bspr_config.json'), 'w', encoding='utf-8') as f:
        json.dump(bspr_config.to_dict(), f, indent=2, sort_keys=True)

experiment_dir_prefix = '{}_v'.format(experiment_config_name)
resume_candidate_dirs = []
for candidate_name in os.listdir(checkpoint_root):
    candidate_path = os.path.join(checkpoint_root, candidate_name)
    if (os.path.isdir(candidate_path)
            and candidate_path != experiment_dir
            and candidate_name.startswith(experiment_dir_prefix)):
        resume_candidate_dirs.append(candidate_path)
resume_candidate_dirs.sort(key=os.path.getmtime, reverse=True)
resume_candidate_dirs.append(checkpoint_root)


def find_resume_file(filename, legacy_filename=None):
    if not args.resume:
        return None
    candidate_filenames = [filename]
    if legacy_filename is not None and legacy_filename != filename:
        candidate_filenames.append(legacy_filename)
    for candidate_dir in resume_candidate_dirs:
        for candidate_filename in candidate_filenames:
            candidate_path = os.path.join(candidate_dir, candidate_filename)
            if os.path.isfile(candidate_path):
                return candidate_path
    return None


log_name = 'A_log_v{}.txt'.format(run_timestamp)
log_path = os.path.join(experiment_dir, log_name)
log_file = open(log_path, 'a', buffering=1, encoding='utf-8')
sys.stdout = TeeStream(sys.stdout, log_file)
sys.stderr = TeeStream(sys.stderr, log_file)

print('Experiment directory: {}'.format(os.path.basename(experiment_dir)))
print('Console log: {}'.format(log_name))
anonymous_arguments = vars(args).copy()
anonymous_arguments['ckp_prefix'] = experiment_config_name
print(json.dumps(anonymous_arguments, sort_keys=True))
if args.deterministic_training:
    print('Deterministic training: enabled; class_order_seed={}, '
          'training_seed={}, requested_policy={}, effective_policy={}, '
          'stage_seed_scheme=splitmix64_run_stage_stream_v1'
          .format(args.random_seed, effective_training_seed,
                  args.deterministic_algorithm_policy,
                  deterministic_algorithm_policy_effective))
else:
    print('Deterministic training: disabled; legacy RNG behavior retained')
if bspr_config.enabled:
    print('Resolved BSPR configuration: {}'.format(
        json.dumps(bspr_config.to_dict(), sort_keys=True)))
base50_checkpoint_path = os.path.join(checkpoint_root, 'Base50_Trained_model.pth')
if args.use_base50_checkpoint:
    if args.nb_cl_fg != 50:
        raise ValueError('--use_base50_checkpoint requires --nb_cl_fg 50.')
    if args.nb_runs != 1:
        raise ValueError(
            '--use_base50_checkpoint requires --nb_runs 1 because one Base50 '
            'model only matches one class order.')
    if not os.path.isfile(base50_checkpoint_path):
        raise FileNotFoundError(
            'Base50 checkpoint not found: Base50_Trained_model.pth')
    print('Base50 checkpoint: Base50_Trained_model.pth')
data_loader_workers = 0 if os.name == 'nt' else 2
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform_test)

dictionary_size     = 500
top1_acc_list_cumul = np.zeros((int(NUM_CLASSES/args.nb_cl),3,args.nb_runs))
top1_acc_list_ori   = np.zeros((int(NUM_CLASSES/args.nb_cl),3,args.nb_runs))

X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)
Y_valid_total = np.array(testset.targets)

for iteration_total in range(args.nb_runs):
    bspr_state = None
    if bspr_config.enabled:
        bspr_state = BSPRState(bspr_config.stable_hash(), device=device)
    order_filename = "seed_{}_{}_order_run_{}.pkl".format(
        args.random_seed, DATASET_NAME, iteration_total)
    order_name = os.path.join(experiment_dir, order_filename)
    resume_order_name = find_resume_file(order_filename)
    print("Order name:{}".format(order_filename))
    if resume_order_name is not None:
        print("Loading orders from {}".format(
            os.path.basename(resume_order_name)))
        order = utils_pytorch.unpickle(resume_order_name)
        utils_pytorch.savepickle(order, order_name)
    else:
        print("Generating orders")
        order = np.arange(NUM_CLASSES)
        if args.deterministic_training:
            class_order_rng.shuffle(order)
        else:
            np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    X_valid_cumuls    = []
    X_protoset_cumuls = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    alpha_dr_herding  = np.zeros((int(NUM_CLASSES/args.nb_cl),dictionary_size,args.nb_cl),np.float32)

    prototypes = np.zeros(
        (NUM_CLASSES, dictionary_size, X_train_total.shape[1],
         X_train_total.shape[2], X_train_total.shape[3]),
        dtype=np.uint8)
    for orde in range(NUM_CLASSES):
        prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]

    start_iter = int(args.nb_cl_fg/args.nb_cl)-1
    for iteration in range(start_iter, int(NUM_CLASSES/args.nb_cl)):
        if args.deterministic_training:
            model_stage_seed = derive_stage_seed(
                effective_training_seed, iteration_total, iteration, 0)
            seed_runtime_random_generators(model_stage_seed)
        else:
            model_stage_seed = None
        if iteration == start_iter:
            last_iter = 0
            tg_model = modified_resnet_cifar.resnet32(num_classes=args.nb_cl_fg)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
        elif iteration == start_iter+1:
            last_iter = iteration
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = out_features*1.0 / args.nb_cl
        else:
            last_iter = iteration
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                out_features1, "out_features2:", out_features2)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (args.nb_cl)

        if iteration > start_iter and args.adapt_lamda:
            cur_lamda = args.lamda * math.sqrt(lamda_mult)
        else:
            cur_lamda = args.lamda
        if iteration > start_iter:
            print("Lamda for less forget is set to ", cur_lamda)

        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10 = np.array([
            i in order[range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl)]
            for i in Y_valid_total
        ])

        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)

        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)

        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            if args.rs_ratio > 0:
                scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
                rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
                rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
                print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
            X_train    = np.concatenate((X_train,X_protoset),axis=0)
            Y_train    = np.concatenate((Y_train,Y_protoset))

        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

        if iteration > start_iter and args.imprint_weights:
            print("Imprint weights")
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features
            novel_embedding = torch.zeros((args.nb_cl, num_features))
            for cls_idx in range(iteration*args.nb_cl, (iteration+1)*args.nb_cl):
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                evalset.data = X_train[cls_indices].astype('uint8')
                evalset.targets = np.zeros(evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=data_loader_workers)
                num_samples = evalset.data.shape[0]
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            tg_model.to(device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(device)

        if args.deterministic_training:
            augmentation_seed = derive_stage_seed(
                effective_training_seed, iteration_total, iteration, 1)
            loader_seed = derive_stage_seed(
                effective_training_seed, iteration_total, iteration, 2)
            sampler_seed = derive_stage_seed(
                effective_training_seed, iteration_total, iteration, 3)
            evaluation_loader_seed = derive_stage_seed(
                effective_training_seed, iteration_total, iteration, 4)
            seed_runtime_random_generators(augmentation_seed)
            print('[Determinism][run {}, stage {}] model_seed={}, '
                  'augmentation_seed={}, loader_seed={}, sampler_seed={}'
                  .format(iteration_total + 1, iteration - start_iter + 1,
                          model_stage_seed, augmentation_seed, loader_seed,
                          sampler_seed))
        else:
            loader_seed = None
            sampler_seed = None
            evaluation_loader_seed = None
        trainset.data = X_train.astype('uint8')
        trainset.targets = map_Y_train
        if iteration > start_iter and args.rs_ratio > 0 and scale_factor > 1:
            print("Weights from sampling:", rs_sample_weights)
            index1 = np.where(rs_sample_weights>1)[0]
            index2 = np.where(map_Y_train<iteration*args.nb_cl)[0]
            assert((index1==index2).all())
            train_sampler = make_weighted_random_sampler(
                rs_sample_weights, rs_num_samples, sampler_seed)
            trainloader = make_seeded_data_loader(trainset,
                deterministic_seed=loader_seed, batch_size=train_batch_size,
                shuffle=False, sampler=train_sampler,
                num_workers=data_loader_workers)
        else:
            trainloader = make_seeded_data_loader(trainset,
                deterministic_seed=loader_seed, batch_size=train_batch_size,
                shuffle=True, num_workers=data_loader_workers)
        testset.data = X_valid_cumul.astype('uint8')
        testset.targets = map_Y_valid_cumul
        testloader = make_seeded_data_loader(testset,
            deterministic_seed=evaluation_loader_seed, batch_size=test_batch_size,
            shuffle=False, num_workers=data_loader_workers)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
        ckp_filename = 'run_{}_iteration_{}_model.pth'.format(
            iteration_total, iteration)
        legacy_ckp_filename = '{}_{}'.format(args.ckp_prefix, ckp_filename)
        ckp_name = os.path.join(experiment_dir, ckp_filename)
        resume_ckp_name = find_resume_file(ckp_filename, legacy_ckp_filename)
        bspr_state_filename = None
        bspr_state_name = None
        if bspr_config.enabled:
            bspr_state_filename = companion_filename(
                '', iteration_total, iteration)
            bspr_state_name = os.path.join(experiment_dir, bspr_state_filename)
        print('ckp_name', ckp_filename)
        stage_number = iteration - start_iter + 1
        if args.use_base50_checkpoint and iteration == start_iter:
            training_flow_message = (
                'not called (the Base50 checkpoint skips initial training)')
        elif resume_ckp_name is not None:
            training_flow_message = (
                'not called (this stage is loaded from a resume checkpoint)')
        else:
            training_flow_message = get_incremental_training_file(
                bspr_config.enabled, incremental_stage=(iteration > start_iter))
        print('[Training flow][run {}, stage {}] {}'.format(
            iteration_total + 1, stage_number, training_flow_message))

        if args.use_base50_checkpoint and iteration == start_iter:
            print("Loading the trained Base50 model: Base50_Trained_model.pth")
            tg_model = load_base50_model(
                tg_model, base50_checkpoint_path, device, args.nb_cl_fg)
            print("Skipping the initial 50-class training stage")
            torch.save(tg_model, ckp_name)
        elif resume_ckp_name is not None:
            print("Loading models from checkpoint: {}".format(
                os.path.basename(resume_ckp_name)))
            tg_model = torch.load(resume_ckp_name, map_location=device).to(device)
            torch.save(tg_model, ckp_name)
            if bspr_config.enabled:
                resume_bspr_filename = bspr_state_filename
                if os.path.basename(resume_ckp_name) == legacy_ckp_filename:
                    resume_bspr_filename = companion_filename(
                        args.ckp_prefix, iteration_total, iteration)
                resume_bspr_name = os.path.join(
                    os.path.dirname(resume_ckp_name), resume_bspr_filename)
                if os.path.isfile(resume_bspr_name):
                    _, bspr_metadata = load_bspr_checkpoint(
                        resume_bspr_name,
                        bspr_state,
                        map_location=device,
                        strict_config=True)
                    validate_bspr_checkpoint_identity(
                        bspr_state,
                        bspr_metadata,
                        expected_run=iteration_total,
                        expected_iteration=iteration,
                        expected_model_checkpoint=resume_ckp_name,
                        num_seen_classes=tg_model.fc.out_features,
                        num_new_classes=args.nb_cl,
                        incremental_stage=(iteration > start_iter))
                    print('Loaded BSPR companion state: {}'.format(
                        os.path.basename(resume_bspr_name)))
                    print('BSPR resume metadata: {}'.format(bspr_metadata))
                elif iteration > start_iter and not bspr_config.allow_state_rebuild:
                    raise FileNotFoundError(
                        'Incremental checkpoint is missing its BSPR companion '
                        'state. Use --bspr_allow_state_rebuild only for an '
                        'explicit memory-based rebuild.')
                elif iteration > start_iter:
                    print('WARNING: rebuilding missing BSPR state from finalized '
                          'memory; pairwise history and dual trajectories are lost.')
                    bspr_state = BSPRState(
                        bspr_config.stable_hash(), device=device)
                    num_seen_classes = tg_model.fc.out_features
                    num_old_classes = num_seen_classes - args.nb_cl
                    bspr_state.begin_stage(
                        iteration,
                        range(num_old_classes),
                        range(num_old_classes, num_seen_classes))
                    bspr_state.state_rebuilt = True
                else:
                    print('Base-stage checkpoint has no BSPR companion; risk '
                          'references will be initialized after herding.')
        else:
            if iteration > start_iter:
                ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, \
                    tg_model.parameters())
                tg_params =[{'params': base_params, 'lr': base_lr, 'weight_decay': custom_weight_decay}, \
                        {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
            else:
                tg_params = tg_model.parameters()
            tg_model = tg_model.to(device)
            if iteration > start_iter:
                ref_model = ref_model.to(device)
            tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            if bspr_config.enabled and iteration > start_iter:
                from utils_incremental.incremental_train_and_eval_MR_LF_BSPR import (
                    incremental_train_and_eval_MR_LF_BSPR)
                num_old_classes = ref_model.fc.out_features
                num_seen_classes = tg_model.fc.out_features
                bspr_state.begin_stage(
                    iteration,
                    range(num_old_classes),
                    range(num_old_classes, num_seen_classes))
                tg_model = incremental_train_and_eval_MR_LF_BSPR(
                    args.epochs, tg_model, ref_model,
                    tg_optimizer, tg_lr_scheduler,
                    trainloader, testloader,
                    iteration, start_iter,
                    cur_lamda,
                    args.dist, args.K, args.lw_mr,
                    device=device,
                    bspr_config=bspr_config,
                    bspr_state=bspr_state)
            else:
                tg_model = incremental_train_and_eval_MR_LF(
                    args.epochs, tg_model, ref_model,
                    tg_optimizer, tg_lr_scheduler,
                    trainloader, testloader,
                    iteration, start_iter,
                    cur_lamda,
                    args.dist, args.K, args.lw_mr,
                    device=device)
            torch.save(tg_model, ckp_name)

        nb_protos_cl = args.nb_protos
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features
        print('Updating exemplar set...')
        for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
            evalset.data = prototypes[iter_dico].astype('uint8')
            evalset.targets = np.zeros(evalset.data.shape[0])
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=data_loader_workers)
            num_samples = evalset.data.shape[0]            
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)

            mu  = np.mean(D,axis=1)
            index1 = int(iter_dico/args.nb_cl)
            index2 = iter_dico % args.nb_cl
            alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
            w_t = mu
            iter_herding     = 0
            iter_herding_eff = 0
            while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                tmp_t   = np.dot(w_t,D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[index1,ind_max,index2] == 0:
                    alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                    iter_herding += 1
                w_t = w_t+mu-D[:,ind_max]

        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = np.zeros((num_features, NUM_CLASSES, 2))
        for iteration2 in range(iteration+1):
            for iter_dico in range(args.nb_cl):
                current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]

                evalset.data = prototypes[iteration2*args.nb_cl+iter_dico].astype('uint8')
                evalset.targets = np.zeros(evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=data_loader_workers)
                num_samples = evalset.data.shape[0]
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                evalset.data = prototypes[iteration2*args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=data_loader_workers)
                mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                D2 = mapped_prototypes2.T
                D2 = D2/np.linalg.norm(D2,axis=0)

                alph = alpha_dr_herding[iteration2,:,iter_dico]
                alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                selected_indices = np.where(alph==1)[0]
                X_protoset_cumuls.append(
                    prototypes[iteration2*args.nb_cl+iter_dico, selected_indices])
                Y_protoset_cumuls.append(
                    order[iteration2*args.nb_cl+iter_dico]
                    * np.ones(len(selected_indices)))
                alph = alph/np.sum(alph)
                class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                alph = np.ones(dictionary_size)/dictionary_size
                class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

        if bspr_config.enabled:
            if bspr_state.stage_index == -1 and iteration == start_iter:
                bspr_state.stage_index = iteration
            elif bspr_state.stage_index != iteration:
                raise RuntimeError(
                    'BSPR state stage {} does not match completed iteration {}.'
                    .format(bspr_state.stage_index, iteration))
            first_stage_class_ids = list(range(
                last_iter * args.nb_cl,
                (iteration + 1) * args.nb_cl))
            reference_candidate_ids = (
                list(range((iteration + 1) * args.nb_cl))
                if bspr_state.state_rebuilt else first_stage_class_ids
            )
            pending_reference_ids = [
                class_id for class_id in reference_candidate_ids
                if (class_id >= int(bspr_state.risk_initialized.numel())
                    or not bool(bspr_state.risk_initialized[class_id].item()))
            ]
            reference_exemplar_count = 0
            if pending_reference_ids:
                reference_images = np.concatenate([
                    X_protoset_cumuls[class_id]
                    for class_id in pending_reference_ids
                ]).astype('uint8')
                reference_labels = np.concatenate([
                    np.full(
                        len(X_protoset_cumuls[class_id]),
                        class_id,
                        dtype=np.int64)
                    for class_id in pending_reference_ids
                ])
                reference_exemplar_count = len(reference_labels)
                class_references = evaluate_reference_risks(
                    model=tg_model,
                    raw_uint8_images=reference_images,
                    mapped_labels=reference_labels,
                    train_transform=transform_train,
                    device=device,
                    num_views=bspr_config.reference_views,
                    batch_size=eval_batch_size,
                    seed=(args.random_seed
                          + iteration_total * 100000
                          + iteration * 1000))
                reference_status = bspr_state.set_class_references(
                    class_references, bspr_config.risk_budget_increment)
            else:
                reference_status = {
                    'initialized': 0,
                    'already_initialized': len(reference_candidate_ids),
                }
            print(
                'BSPR risk references: {} exemplars, {} initialized classes, '
                '{} already initialized'.format(
                    reference_exemplar_count,
                    reference_status['initialized'],
                    reference_status['already_initialized']))

            save_bspr_checkpoint(
                bspr_state_name,
                bspr_state,
                bspr_config,
                metadata={
                    'run': iteration_total,
                    'iteration': iteration,
                    'model_checkpoint': os.path.basename(ckp_name),
                    'state_rebuilt': bspr_state.state_rebuilt,
                    'saved_after_herding': True,
                    'reference_classes_first_seen': first_stage_class_ids,
                })
            print('Saved BSPR companion state: {}'.format(
                os.path.basename(bspr_state_name)))

        class_means_name = 'run_{}_iteration_{}_class_means.pth'.format(
            iteration_total, iteration)
        torch.save(class_means, os.path.join(experiment_dir, class_means_name))

        current_means = class_means[:, order[range(0,(iteration+1)*args.nb_cl)]]
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the original batch of classes...')
        evalset.data = X_valid_ori.astype('uint8')
        evalset.targets = map_Y_valid_ori
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=data_loader_workers)
        ori_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        print('Computing cumulative accuracy...')
        evalset.data = X_valid_cumul.astype('uint8')
        evalset.targets = map_Y_valid_cumul
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=data_loader_workers)        
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
    top1_ori_name = 'run_{}_top1_acc_list_ori.pth'.format(iteration_total)
    top1_cumul_name = 'run_{}_top1_acc_list_cumul.pth'.format(iteration_total)
    torch.save(top1_acc_list_ori, os.path.join(experiment_dir, top1_ori_name))
    torch.save(top1_acc_list_cumul, os.path.join(experiment_dir, top1_cumul_name))

print_cumulative_accuracy_summary(
    top1_acc_list_cumul,
    start_iteration=int(args.nb_cl_fg/args.nb_cl)-1,
    classes_per_iteration=args.nb_cl)
