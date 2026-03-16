import torch


def collect_outputs(models, loader, device):
    all_probs = []

    for model in models:
        model.eval()
        probs = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                out = torch.softmax(model(x), dim=1)
                probs.append(out.cpu())

        all_probs.append(torch.cat(probs))

    return torch.stack(all_probs) 


def soft_voting_cached(probs):
    return probs.mean(dim=0).argmax(dim=1)


def hard_voting_cached(probs):
    preds = probs.argmax(dim=2)
    return preds.mode(dim=0).values


def weighted_soft_voting_cached(probs, weights):
    w = torch.tensor(weights).view(-1, 1, 1)
    weighted = probs * w
    return weighted.sum(dim=0).argmax(dim=1)