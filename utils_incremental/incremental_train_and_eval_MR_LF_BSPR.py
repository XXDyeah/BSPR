import torch
import torch.nn as nn

from bspr.deterministic_ops import margin_ranking_loss_without_scatter
from bspr.plugin import BSPRPlugin


def _mean(total, count):
    return total / max(count, 1)


def incremental_train_and_eval_MR_LF_BSPR(
        epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
        trainloader, testloader,
        iteration, start_iteration,
        lamda, dist, K, lw_mr,
        device=None,
        bspr_config=None, bspr_state=None):
    if iteration <= start_iteration:
        raise ValueError("BSPR requires an incremental stage")
    if bspr_config is None or bspr_state is None or not bspr_config.enabled:
        raise ValueError("an enabled BSPR configuration and state are required")
    if ref_model is None:
        raise ValueError("the reference model is required")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if not hasattr(tg_model, "forward_with_features_and_angles"):
        raise TypeError("target model lacks the feature and angle interface")
    if not hasattr(ref_model, "forward_with_features"):
        raise TypeError("reference model lacks the feature interface")
    if not hasattr(tg_model.fc, "normalized_weight_matrix"):
        raise TypeError("classifier head lacks normalized class directions")

    ref_model.eval()
    num_old_classes = int(ref_model.fc.out_features)
    plugin = BSPRPlugin(bspr_config, bspr_state, tg_model.fc)
    ce_criterion = nn.CrossEntropyLoss()
    lf_criterion = nn.CosineEmbeddingLoss()

    for epoch in range(int(epochs)):
        tg_model.train()
        print("\nEpoch: {}, LR: {}".format(epoch, tg_lr_scheduler.get_last_lr()))

        totals = {
            "lf": 0.0,
            "ce": 0.0,
            "mr": 0.0,
            "base": 0.0,
            "probe": 0.0,
            "risk": 0.0,
            "addon": 0.0,
            "loss": 0.0,
            "accepted": 0.0,
        }
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
            with torch.no_grad():
                _, reference_features = ref_model.forward_with_features(inputs)

            loss_lf = lf_criterion(
                features,
                reference_features,
                torch.ones(inputs.size(0), device=device),
            ) * float(lamda)
            loss_ce = ce_criterion(outputs, targets)
            loss_mr = margin_ranking_loss_without_scatter(
                angular_scores,
                targets,
                num_old_classes,
                dist,
                K,
                lw_mr,
            )
            base_loss = loss_lf + loss_ce + loss_mr

            step = plugin.compute_losses(
                features,
                outputs,
                angular_scores,
                targets,
                num_old_classes,
            )
            probe_loss = float(bspr_config.probe_coefficient) * step.probe_loss
            risk_loss = (
                float(bspr_config.risk_coefficient)
                * step.old_fraction
                * step.risk_loss
            )
            addon_loss = probe_loss + risk_loss
            loss = base_loss + addon_loss

            loss.backward()
            tg_optimizer.step()
            plugin.commit_post_step_statistics(step.delayed_statistics)

            accepted = step.accepted.detach()
            totals["lf"] += float(loss_lf.detach().item())
            totals["ce"] += float(loss_ce.detach().item())
            totals["mr"] += float(loss_mr.detach().item())
            totals["base"] += float(base_loss.detach().item())
            totals["probe"] += float(probe_loss.detach().item())
            totals["risk"] += float(risk_loss.detach().item())
            totals["addon"] += float(addon_loss.detach().item())
            totals["loss"] += float(loss.detach().item())
            totals["accepted"] += float(accepted.item())
            correct += int(outputs.detach().argmax(dim=1).eq(targets).sum().item())
            sample_count += int(targets.size(0))
            batch_count += 1

        tg_lr_scheduler.step()

        print(
            "Train set: {}, LF: {:.4f}, CE: {:.4f}, MR: {:.4f}, "
            "Base: {:.4f}, Loss: {:.4f}, Acc: {:.4f}".format(
                len(trainloader),
                _mean(totals["lf"], batch_count),
                _mean(totals["ce"], batch_count),
                _mean(totals["mr"], batch_count),
                _mean(totals["base"], batch_count),
                _mean(totals["loss"], batch_count),
                100.0 * correct / max(sample_count, 1),
            )
        )
        print(
            "BSPR: Probe: {:.6f}, Risk: {:.6f}, Add-on: {:.6f}, "
            "Accepted probes/batch: {:.3f}".format(
                _mean(totals["probe"], batch_count),
                _mean(totals["risk"], batch_count),
                _mean(totals["addon"], batch_count),
                _mean(totals["accepted"], batch_count),
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
