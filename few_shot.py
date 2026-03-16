import torch
import torch.nn as nn
import torch.nn.functional as F
import random




class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return F.normalize(x, dim=1)




def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    elif hasattr(dataset, "indices"):
        return [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        raise ValueError("Unsupported dataset")




def create_episode(dataset, n_way, k_shot, q_query):
    labels = get_labels(dataset)

    class_to_indices = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)

    selected_classes = random.sample(list(class_to_indices.keys()), n_way)

    support_idx = []
    query_idx = []

    for i, c in enumerate(selected_classes):
        indices = class_to_indices[c]
        random.shuffle(indices)

        support = indices[:k_shot]
        query = indices[k_shot:k_shot + q_query]

        support_idx.extend(support)
        query_idx.extend(query)

    return support_idx, query_idx



def compute_prototypes(embeddings, labels, n_way):
    prototypes = []
    for c in range(n_way):
        proto = embeddings[labels == c].mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)


def prototypical_loss(model, support_x, support_y, query_x, query_y, n_way):
    support_emb = model(support_x)
    query_emb = model(query_x)

    prototypes = compute_prototypes(support_emb, support_y, n_way)

    dists = torch.cdist(query_emb, prototypes)
    logits = -dists

    loss = F.cross_entropy(logits, query_y)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_y).float().mean()

    return loss, acc




def train_protonet(model, dataset, optimizer, device,
                   epochs=30, n_way=5, k_shot=5, q_query=5, episodes_per_epoch=100):

    logs = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_acc = 0

        for _ in range(episodes_per_epoch):
            s_idx, q_idx = create_episode(dataset, n_way, k_shot, q_query)

            support = [dataset[i] for i in s_idx]
            query = [dataset[i] for i in q_idx]

            support_x = torch.stack([x for x, _ in support]).to(device)
            query_x = torch.stack([x for x, _ in query]).to(device)


            support_y = torch.tensor(
                [i // k_shot for i in range(len(support))],
                device=device
            )
            query_y = torch.tensor(
                [i // q_query for i in range(len(query))],
                device=device
            )

            optimizer.zero_grad()

            loss, acc = prototypical_loss(
                model, support_x, support_y, query_x, query_y, n_way
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        logs["loss"].append(total_loss / episodes_per_epoch)
        logs["acc"].append(total_acc / episodes_per_epoch)

        print(f"Epoch {epoch+1}: loss={logs['loss'][-1]:.4f}, acc={logs['acc'][-1]:.4f}")

    return logs



def evaluate_protonet(model, dataset, device,
                      n_way=5, k_shot=5, q_query=5, episodes=100):

    model.eval()
    total_acc = 0

    with torch.no_grad():
        for _ in range(episodes):
            s_idx, q_idx = create_episode(dataset, n_way, k_shot, q_query)

            support = [dataset[i] for i in s_idx]
            query = [dataset[i] for i in q_idx]

            support_x = torch.stack([x for x, _ in support]).to(device)
            query_x = torch.stack([x for x, _ in query]).to(device)

            support_y = torch.tensor([i // k_shot for i in range(len(support))], device=device)
            query_y = torch.tensor([i // q_query for i in range(len(query))], device=device)

            _, acc = prototypical_loss(
                model, support_x, support_y, query_x, query_y, n_way
            )

            total_acc += acc.item()

    return total_acc / episodes