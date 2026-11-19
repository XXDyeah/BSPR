import torch
import torch.nn as nn

from bspr.deterministic_ops import margin_ranking_loss_without_scatter


def _mean(total, count):
    return total / max(count, 1)


def incremental_train_and_eval_MR_LF(
        epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
        trainloader, testloader,
        iteration, start_iteration,
        lamda, dist, K, lw_mr,
        device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    incremental_stage = iteration > start_iteration
    if not hasattr(tg_model, "forward_with_features_and_angles"):
        raise TypeError("target model lacks the feature and angle interface")
    if incremental_stage:
        if ref_model is None or not hasattr(ref_model, "forward_with_features"):
            raise TypeError("reference model lacks the feature interface")
        ref_model.eval()
        num_old_classes = int(ref_model.fc.out_features)

    ce_criterion = nn.CrossEntropyLoss()
    lf_criterion = nn.CosineEmbeddingLoss()

    for epoch in range(int(epochs)):
        tg_model.train()
        print("\nEpoch: {}, LR: {}".format(epoch, tg_lr_scheduler.get_last_lr()))

        totals = {"lf": 0.0, "ce": 0.0, "mr": 0.0, "loss": 0.0}
        correct = 0
        sample_count = 0
        batch_count = 0

        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            tg_optimizer.zero_grad()

            outputs, features, angular_scores = (
                tg_model.forward_with_features_and_angles(inputs)
            )
            loss_ce = ce_criterion(outputs, targets)
            loss_lf = outputs.new_zeros(())
            loss_mr = outputs.new_zeros(())

            if incremental_stage:
                with torch.no_grad():
                    _, reference_features = ref_model.forward_with_features(inputs)
                loss_lf = lf_criterion(
                    features,
                    reference_features,
                    torch.ones(inputs.size(0), device=device),
                ) * float(lamda)
                loss_mr = margin_ranking_loss_without_scatter(
                    angular_scores,
                    targets,
                    num_old_classes,
                    dist,
                    K,
                    lw_mr,
                )

            loss = loss_ce + loss_lf + loss_mr
            loss.backward()
            tg_optimizer.step()

            totals["lf"] += float(loss_lf.detach().item())
            totals["ce"] += float(loss_ce.detach().item())
            totals["mr"] += float(loss_mr.detach().item())
            totals["loss"] += float(loss.detach().item())
            correct += int(outputs.detach().argmax(dim=1).eq(targets).sum().item())
            sample_count += int(targets.size(0))
            batch_count += 1

        tg_lr_scheduler.step()

        print(
            "Train set: {}, LF: {:.4f}, CE: {:.4f}, MR: {:.4f}, "
            "Loss: {:.4f}, Acc: {:.4f}".format(
                len(trainloader),
                _mean(totals["lf"], batch_count),
                _mean(totals["ce"], batch_count),
                _mean(totals["mr"], batch_count),
                _mean(totals["loss"], batch_count),
                100.0 * correct / max(sample_count, 1),
            )
        )

        tg_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_count = 0
        test_batches = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = tg_model(inputs)
                test_loss += float(ce_criterion(outputs, targets).item())
                test_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
                test_count += int(targets.size(0))
                test_batches += 1
        print(
            "Test set: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                len(testloader),
                _mean(test_loss, test_batches),
                100.0 * test_correct / max(test_count, 1),
            )
        )

    return tg_model
