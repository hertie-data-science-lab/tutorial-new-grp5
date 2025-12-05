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
    false_negatives = defaultdict(int)
    # For demographic parity
    positive_preds = defaultdict(int)
    # For individual fairness proxy
    embeddings = defaultdict(list)   # group → list of embeddings
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Extract embeddings before final FC layer
            if hasattr(model, "fc"):
                feat = model.avgpool(outputs).squeeze() if len(outputs.shape) == 4 else outputs
            else:
                feat = outputs
            
            for img_emb, t, p in zip(feat, labels, preds):
                true_class = classes[t.item()]
                pred_class = classes[p.item()]
                
                # General stats
                group_total[true_class] += 1
                if t.item() == p.item():
                    group_correct[true_class] += 1
                    true_positives[pred_class] += 1
                else:
                    false_positives[pred_class] += 1
                    false_negatives[true_class] += 1
                
                # Demographic parity tracking
                if p.item() == 1:  # positive prediction; adjust if your positive label differs
                    positive_preds[true_class] += 1
                
                # Embeddings for individual fairness
                embeddings[true_class].append(img_emb.cpu().numpy())
    
    # --- Accuracy metrics ---
    overall_accuracy = sum(group_correct.values()) / sum(group_total.values()) * 100
    per_class_acc = {
        c: (group_correct[c] / group_total[c] * 100) if group_total[c] else None
        for c in classes
    }
    acc_values = [v for v in per_class_acc.values() if v is not None]
    acc_std = np.std(acc_values) if len(acc_values) > 1 else None
    
    # --- Precision (Predictive Parity) ---
    precision = {
        c: (true_positives[c] / (true_positives[c] + false_positives[c]) * 100)
        if (true_positives[c] + false_positives[c]) else None
        for c in classes
    }
    precision_values = [p for p in precision.values() if p is not None]
    macro_precision = sum(precision_values) / len(precision_values) if precision_values else None
    
    # --- Recall (for F1 calculation) ---
    recall = {
        c: (true_positives[c] / (true_positives[c] + false_negatives[c]) * 100)
        if (true_positives[c] + false_negatives[c]) else None
        for c in classes
    }
    
    # --- F1 Score ---
    f1_score = {
        c: (2 * (precision[c] / 100) * (recall[c] / 100) / 
            ((precision[c] / 100) + (recall[c] / 100)) * 100)
        if precision[c] is not None and recall[c] is not None 
           and (precision[c] + recall[c]) > 0
        else None
        for c in classes
    }
    f1_values = [f for f in f1_score.values() if f is not None]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else None
    
    # --- Demographic Parity ---
    demographic_parity = {
        c: (positive_preds[c] / group_total[c]) if group_total[c] else None
        for c in classes
    }
    
    # --- Equal Opportunity (TPR parity) ---
    tpr = {
        c: (true_positives[c] / (true_positives[c] + false_negatives[c]))
        if (true_positives[c] + false_negatives[c]) else None
        for c in classes
    }
    
    # --- Individual Fairness (Proxy metric) ---
    # Compute avg pairwise distance within each class
    individual_fairness = {}
    for c in classes:
        vecs = embeddings[c]
        if len(vecs) < 2:
            individual_fairness[c] = None
        else:
            dists = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    dists.append(np.linalg.norm(vecs[i] - vecs[j]))
            individual_fairness[c] = np.mean(dists)
    
    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_acc,
        "accuracy_std_dev": acc_std,
        # Precision metrics
        "precision": precision,
        "macro_precision": macro_precision,
        # Recall metrics
        "recall": recall,
        # F1 Score metrics
        "f1_score": f1_score,
        "macro_f1": macro_f1,
        # Fairness metrics
        "demographic_parity": demographic_parity,
        "TPR_equal_opportunity": tpr,
        "individual_fairness_proxy": individual_fairness,
    }

def summarize_model(model):
    return summary(model, input_size=(1, 3, 224, 224))

# Helper function to reverse the normalization for plotting
def denormalize_image(tensor):
    """Denormalizes a single (C, H, W) tensor to (H, W, C) numpy array [0, 1]."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_tensor = torch.from_numpy(tensor) if isinstance(tensor, np.ndarray) else tensor.cpu()
    
    denormalized_img = img_tensor * STD + MEAN
    # Convert from (C, H, W) to (H, W, C) for matplotlib
    denormalized_img = denormalized_img.permute(1, 2, 0).numpy()
    return np.clip(denormalized_img, 0, 1) # Clip to valid range [0, 1]
