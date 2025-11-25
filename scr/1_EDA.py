from imports import *

# ----------------------------------------------------
# 1. Load dataset structure
# ----------------------------------------------------
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes detected:")
for c in classes:
    print(" -", c)

# Gather image paths
image_paths = []
labels = []

for cls in classes:  
    cls_dir = os.path.join(DATA_DIR, cls)

    # now iterate subfolders inside each class
    for identity in os.listdir(cls_dir):
        id_dir = os.path.join(cls_dir, identity)
        
        # ensure it's a folder
        if not os.path.isdir(id_dir):
            continue
        
        # now read image files from inside identity folder
        for fname in os.listdir(id_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                image_paths.append(os.path.join(id_dir, fname))
                labels.append(cls)

# ----------------------------------------------------
# 2. Class distribution
# ----------------------------------------------------
class_counts = Counter(labels)
print("\nClass distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")


# ----------------------------------------------------
# 3. Image dimension statistics
# ----------------------------------------------------
widths = []
heights = []


for path in image_paths[:500]: # sample first 500 images for speed
    try:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass


print(f"\nSampled {len(widths)} images for dimension analysis.")
print(f"Average width: {sum(widths)/len(widths):.1f}")
print(f"Average height: {sum(heights)/len(heights):.1f}")


# ----------------------------------------------------
# 4. Plot one random image per class
# ----------------------------------------------------
unique_classes = sorted(set(labels))

plt.figure(figsize=(15, 8))

for i, cls in enumerate(unique_classes):
    # find all image indices belonging to this class
    cls_indices = [idx for idx, lbl in enumerate(labels) if lbl == cls]

    # pick a random index
    rand_idx = random.choice(cls_indices)

    img_path = image_paths[rand_idx]
    img = Image.open(img_path)

    # plot
    plt.subplot(1, len(unique_classes), i + 1)
    plt.imshow(img)
    plt.title(cls, fontsize=10)
    plt.axis("off")

plt.tight_layout()
# SAVE FIGURE
plt.show()

# ----------------------------------------------------
# 5. Plot mean faces per class
# ----------------------------------------------------

# Mean faces repeorduced from https://github.com/visionjo/facerec-bias-bfw/blob/master/code/notebooks/1a_generate_mean_faces.ipynb

font = {'font.family': 'serif',
        'font.serif' : 'Times New Roman',
        'font.color':  'darkred',
        'font.weight': 'normal',
        'font.size': 16,
        }
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style('whitegrid', font)

dir_subgroups = Path(f"{DATA_DIR}").glob('*males')
dir_subgroups =list(dir_subgroups)
dir_subgroups.sort()

fig, axs = plt.subplots(4, 2, figsize=(18, 12))
mean_images = []
for dir_subgroup, ax in tqdm(zip(dir_subgroups, axs.flatten())):
    images = np.array([resize(read(f_image), 124, 108) for f_image in Path(dir_subgroup).glob("*/*.jpg")])
    mean_face = np.array(images).mean(0)
    ax.imshow(mean_face)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(dir_subgroup).split('/')[-1], fontsize=20)
    
    mean_images.append(mean_face)
plt.tight_layout()
# SAVE FIGURE
plt.show()
