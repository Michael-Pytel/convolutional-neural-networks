import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ProtoNet(nn.Module):
    def __init__(self, backbone, feat_dim=512, proj_dim=128):
        super().__init__()

        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.backbone(x)

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.head(x)
        return F.normalize(x, dim=1)




def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets

    elif hasattr(dataset, "samples"):
        return [s[1] for s in dataset.samples]

    elif hasattr(dataset, "indices"):
        return [dataset.dataset.samples[i][1] for i in dataset.indices]

    else:
        raise ValueError("Unsupported dataset")


def fetch_batch(dataset, indices, device):
    xs, ys = zip(*[dataset[i] for i in indices])
    return torch.stack(xs).to(device), torch.tensor(ys, device=device)



def create_episode(dataset, n_way, k_shot, q_query, rng):
    labels = get_labels(dataset)

    class_to_indices = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)


    valid_classes = sorted([
        c for c in class_to_indices
        if len(class_to_indices[c]) >= k_shot + q_query
    ])

    selected_classes = rng.sample(valid_classes, n_way)

    support_idx, query_idx = [], []

    for c in selected_classes:
        indices = list(class_to_indices[c])
        rng.shuffle(indices)

        support_idx += indices[:k_shot]
        query_idx += indices[k_shot:k_shot + q_query]

    return support_idx, query_idx, selected_classes


def create_fixed_episodes(dataset, n_way, k_shot, q_query, n_episodes, seed=42):
    rng = random.Random(seed)

    episodes = [
        create_episode(dataset, n_way, k_shot, q_query, rng)
        for _ in range(n_episodes)
    ]

    return episodes


def save_episodes(episodes, path):
    torch.save(episodes, path)


def load_episodes(path):
    return torch.load(path)




def prototypical_step(model, support_x, support_y, query_x, query_y, n_way):
    support_emb = model(support_x)
    query_emb = model(query_x)


    prototypes = torch.stack([
        support_emb[support_y == c].mean(dim=0)
        for c in range(n_way)
    ])

    logits = torch.matmul(query_emb, prototypes.T)

    temp = torch.clamp(model.temperature, 0.01, 10)
    logits = logits / temp

    loss = F.cross_entropy(logits, query_y)

    preds = logits.argmax(dim=1)
    acc = (preds == query_y).float().mean()

    return loss, acc




def evaluate_fixed(model, dataset, episodes, device, n_way):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for s_idx, q_idx, classes in episodes:

            support_x, support_y_raw = fetch_batch(dataset, s_idx, device)
            query_x, query_y_raw = fetch_batch(dataset, q_idx, device)

            class_map = {c: i for i, c in enumerate(classes)}

            support_y = torch.tensor(
                [class_map[y.item()] for y in support_y_raw],
                device=device
            )
            query_y = torch.tensor(
                [class_map[y.item()] for y in query_y_raw],
                device=device
            )

            loss, acc = prototypical_step(
                model, support_x, support_y, query_x, query_y, n_way
            )

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(episodes), total_acc / len(episodes)




def train_protonet(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    device,
    epochs=20,
    n_way=5,
    k_shot=5,
    q_query=5,
    episodes_per_epoch=100,
    seed=42,
    early_stopping=None,
    scheduler=None
):
    model.to(device)


    train_episodes = create_fixed_episodes(
        train_dataset,
        n_way,
        k_shot,
        q_query,
        n_episodes=epochs * episodes_per_epoch,
        seed=seed
    )


    val_episodes = create_fixed_episodes(
        val_dataset,
        n_way,
        k_shot,
        q_query,
        n_episodes=200,
        seed=seed + 1
    )

    logs = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    episode_ptr = 0

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_acc = 0

        for _ in range(episodes_per_epoch):
            s_idx, q_idx, classes = train_episodes[episode_ptr]
            episode_ptr += 1

            support_x, support_y_raw = fetch_batch(train_dataset, s_idx, device)
            query_x, query_y_raw = fetch_batch(train_dataset, q_idx, device)

            class_map = {c: i for i, c in enumerate(classes)}

            support_y = torch.tensor(
                [class_map[y.item()] for y in support_y_raw],
                device=device
            )
            query_y = torch.tensor(
                [class_map[y.item()] for y in query_y_raw],
                device=device
            )

            optimizer.zero_grad()

            loss, acc = prototypical_step(
                model, support_x, support_y, query_x, query_y, n_way
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        train_loss = total_loss / episodes_per_epoch
        train_acc = total_acc / episodes_per_epoch

        val_loss, val_acc = evaluate_fixed(
            model, val_dataset, val_episodes, device, n_way
        )

        logs["train_loss"].append(train_loss)
        logs["train_acc"].append(train_acc)
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.stop:
                break
        
        if scheduler is not None:
            scheduler.step()

    return logs