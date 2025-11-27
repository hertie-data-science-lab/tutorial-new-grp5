from src.imports import *


def load_classes(data_dir):
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    return classes


def load_model(num_classes, model_path, device="cpu"):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


def get_val_loader(data_dir, val_indices_path, batch_size=32):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    val_indices = torch.load(val_indices_path)
    subset = Subset(dataset, val_indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=False), dataset.classes


def evaluate_fairness(model, data_loader, classes, device="cpu"):
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for t, p in zip(labels, preds):
                true_class = classes[t.item()]
                pred_class = classes[p.item()]

                group_total[true_class] += 1
                if t.item() == p.item():
                    group_correct[true_class] += 1
                    true_positives[pred_class] += 1
                else:
                    false_positives[pred_class] += 1

    # Compute metrics
    overall = {
        "correct": sum(group_correct.values()),
        "total": sum(group_total.values())
    }
    overall["accuracy"] = overall["correct"] / overall["total"] * 100

    per_class_acc = {
        c: (group_correct[c] / group_total[c] * 100) if group_total[c] > 0 else None
        for c in classes
    }

    acc_values = [a for a in per_class_acc.values() if a is not None]
    std_dev = np.std(acc_values) if len(acc_values) > 1 else None

    precision = {
        c: (true_positives[c] / (true_positives[c] + false_positives[c]) * 100)
        if (true_positives[c] + false_positives[c]) > 0 else None
        for c in classes
    }

    precision_values = [p for p in precision.values() if p is not None]
    macro_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values else None
    )

    return {
        "overall_accuracy": overall["accuracy"],
        "per_class_accuracy": per_class_acc,
        "accuracy_std_dev": std_dev,
        "precision": precision,
        "macro_precision": macro_precision,
    }


def summarize_model(model):
    return summary(model, input_size=(1, 3, 224, 224))
