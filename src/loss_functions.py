import torch

EPS = 1e-07

def cast_labels(y_true, y_pred):
    return y_true.type(torch.float32), y_pred.type(torch.float32) + EPS

def get_WFL(alpha, gamma):
    """
    Returns the Weighted Focal Loss (WFL)
    Args:
        alpha: weight parameter to compensate class imbalance.
        gamma: defines the rate at which easy examples are down-weighted: 
               the higher ,the wider the range in which an example receives low loss.
    Example:
        alpha = 0.8
        gamma = 2.0
        loss_function = get_WFL(alpha, gamma)
    """
    def WFL(y_pred, y_true):
        if len(y_pred.shape) != len(y_true.shape):
            raise ValueError(f"Shape of y_pred {y_pred.shape} is different than y_true shape {y_true.shape}")
        y_real, y_pred = cast_labels(y_true, y_pred)
        loss = -alpha * torch.pow(1 - y_pred, gamma) * \
            y_true * torch.log(y_pred) - \
            (1 - alpha) * torch.pow(1 - (1 - y_pred), gamma) * \
            (1 - y_true) * torch.log(1 - y_pred)
        loss = torch.mean(loss.type(torch.float32))
        return loss
    return WFL