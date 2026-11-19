import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


_ACCURACY_LABELS = ("CNN", "iCaRL", "NCM")


def _print_accuracies(values):
    for label, value in zip(_ACCURACY_LABELS, values):
        print("  top 1 accuracy {:<15}:\t\t{:.2f} %".format(label, value))


def _nearest_mean_predictions(class_means, features, mean_index, device):
    distances = cdist(
        class_means[:, :, mean_index].T, features, "sqeuclidean")
    return torch.from_numpy((-distances).T).to(device).argmax(dim=1)


def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader,
                     scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tg_model.eval()
    tg_feature_model.eval()
    correct = [0, 0, 0]
    total = 0

    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            total += targets.size(0)

            outputs = F.softmax(tg_model(inputs), dim=1)
            if scale is not None:
                assert scale.shape[0] == 1
                assert outputs.shape[1] == scale.shape[1]
                outputs = outputs / scale.to(
                    device=device, dtype=torch.float32).repeat(outputs.size(0), 1)
            output_features = (
                tg_feature_model(inputs).detach().cpu().numpy()
                .reshape(inputs.size(0), -1)
            )
            predictions = (
                outputs.argmax(dim=1),
                _nearest_mean_predictions(
                    class_means, output_features, 0, device),
                _nearest_mean_predictions(
                    class_means, output_features, 1, device),
            )
            for index, prediction in enumerate(predictions):
                correct[index] += prediction.eq(targets).sum().item()

    accuracies = [100.0 * count / total for count in correct]
    if print_info:
        _print_accuracies(accuracies)
    return accuracies


def print_cumulative_accuracy_summary(top1_acc_list_cumul, start_iteration=0,
                                      classes_per_iteration=None):
    accuracies = np.asarray(top1_acc_list_cumul)
    if accuracies.ndim != 3 or accuracies.shape[1] != 3:
        raise ValueError(
            "top1_acc_list_cumul must have shape (tasks, 3, runs), got {}"
            .format(accuracies.shape))
    if start_iteration < 0 or start_iteration >= accuracies.shape[0]:
        raise ValueError(
            "start_iteration must be in [0, {}), got {}".format(
                accuracies.shape[0], start_iteration))

    print("\n" + "=" * 70)
    print("Cumulative accuracy summary (all incremental tasks)")
    print("=" * 70)
    for run_index in range(accuracies.shape[2]):
        if accuracies.shape[2] > 1:
            print("Run {}/{}".format(run_index + 1, accuracies.shape[2]))
        for task_index, iteration in enumerate(
                range(start_iteration, accuracies.shape[0]), start=1):
            description = "Task {}".format(task_index)
            if classes_per_iteration is not None:
                description += " ({} classes)".format(
                    (iteration + 1) * classes_per_iteration)
            print("\n{}".format(description))
            print("Computing cumulative accuracy...")
            _print_accuracies(accuracies[iteration, :, run_index])

        print("\nAverage accuracy over all tasks:")
        for label, column in (("Avg_CNN", 0), ("AVG_NCM", 2)):
            mean = np.mean(accuracies[start_iteration:, column, run_index])
            print("  {:<30}:\t\t{:.2f} %".format(label, mean))
        if run_index + 1 < accuracies.shape[2]:
            print("\n" + "-" * 70)
    print("=" * 70)
