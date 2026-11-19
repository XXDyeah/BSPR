import numpy as np
import torch


def compute_features(tg_feature_model, evalloader, num_samples, num_features,
                     device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tg_feature_model.eval()
    features = np.zeros((num_samples, num_features))
    start = 0

    with torch.no_grad():
        for inputs, _ in evalloader:
            batch_size = inputs.size(0)
            batch_features = (
                tg_feature_model(inputs.to(device))
                .detach().cpu().numpy().reshape(batch_size, num_features)
            )
            features[start:start + batch_size] = batch_features
            start += batch_size

    assert start == num_samples
    return features
