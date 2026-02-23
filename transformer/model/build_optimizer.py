import torch.optim

def build_optimizer(model,opt_name="adam",lr=1e-3,weight_decay=0,momentum=0.9):
    opt_name = opt_name.lower()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
    raise ValueError(f"Unknown optimizer:{opt_name}")