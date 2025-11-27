from imports import *

classes = sorted([
    d for d in os.listdir(DATA_DIR) 
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

print("Classes detected:")
for c in classes:
    print(" -", c)

num_classes = len(classes)
print("Number of classes:", num_classes)

# 1. Recreate the architecture
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 2. Load weights (correct path)
model_path = os.path.join("face_classifier_resnet18.pth")
state = torch.load(model_path, map_location="cpu")

# 3. Assign weights
model.load_state_dict(state)

# 4. Put in evaluation mode
model.eval()
print("Model loaded successfully.")

# model structure summary 
summary(model, input_size=(1, 3, 224, 224))

# Basic fairness assessment 
group_correct = defaultdict(int)
group_total = defaultdict(int)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
val_indices = torch.load('val_indices.pt')

val_subset = Subset(val_dataset, val_indices)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

group_correct = defaultdict(int)
group_total = defaultdict(int)
true_positives = defaultdict(int)
false_positives = defaultdict(int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        for t, p in zip(labels, preds):
            true_class = classes[t.item()]
            pred_class = classes[p.item()]
            
            # For accuracy
            group_total[true_class] += 1
            if t.item() == p.item():
                group_correct[true_class] += 1
                true_positives[pred_class] += 1
            else:
                false_positives[pred_class] += 1

print("\n=== Fairness / Bias Assessment ===")
overall_correct = sum(group_correct.values())
overall_total = sum(group_total.values())
overall_acc = overall_correct / overall_total * 100
print(f"Overall accuracy: {overall_acc:.2f}%\n")

# Add this section to print per-group accuracy
print("Per-class accuracy:")
for class_name in classes:
    if group_total[class_name] > 0:
        class_acc = (group_correct[class_name] / group_total[class_name]) * 100
        print(f"  {class_name}: {class_acc:.2f}% ({group_correct[class_name]}/{group_total[class_name]})")
    else:
        print(f"  {class_name}: No samples in validation set")

# Optional: Calculate and show standard deviation to measure fairness
accuracies = [
    (group_correct[c] / group_total[c] * 100) 
    for c in classes if group_total[c] > 0
]
if len(accuracies) > 1:
    std_dev = np.std(accuracies)
    print(f"\nAccuracy standard deviation: {std_dev:.2f}%")
    print("(Lower std dev = more fair/balanced performance)")

    print("\n=== Per-Class Precision ===")
print("Precision = TP / (TP + FP)\n")
for class_name in classes:
    tp = true_positives[class_name]
    fp = false_positives[class_name]
    
    if (tp + fp) > 0:
        precision = (tp / (tp + fp)) * 100
        print(f"  {class_name}: {precision:.2f}% (TP={tp}, FP={fp})")
    else:
        print(f"  {class_name}: No predictions made")

# Optional: Average precision across all classes
precisions = [
    (true_positives[c] / (true_positives[c] + false_positives[c]) * 100)
    for c in classes if (true_positives[c] + false_positives[c]) > 0
]
if precisions:
    avg_precision = sum(precisions) / len(precisions)
    print(f"\nMacro-averaged precision: {avg_precision:.2f}%")