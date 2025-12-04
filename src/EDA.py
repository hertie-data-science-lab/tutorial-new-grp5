from src.imports import *

class FaceDatasetAnalyzer:
    """
    Analyzer for face recognition datasets with hierarchical structure:
    DATA_DIR/class/identity/images
    """
    
    def __init__(self, data_dir):
        """
        Initialize the analyzer with a data directory.
        
        Args:
            data_dir: Path to the root directory containing class folders
        """
        self.data_dir = data_dir
        self.classes = []
        self.image_paths = []
        self.labels = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset structure and gather all image paths with labels."""
        # Find all class directories
        self.classes = sorted([
            d for d in os.listdir(self.data_dir) 
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        
        # Gather image paths and labels
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            
            # Iterate through identity subfolders
            for identity in os.listdir(cls_dir):
                id_dir = os.path.join(cls_dir, identity)
                
                if not os.path.isdir(id_dir):
                    continue
                
                # Read image files from identity folder
                for fname in os.listdir(id_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                        self.image_paths.append(os.path.join(id_dir, fname))
                        self.labels.append(cls)
    
    def print_classes(self):
        """Print detected classes."""
        print("Classes detected:")
        for c in self.classes:
            print(f"  - {c}")
    
    def get_class_distribution(self):
        """
        Get class distribution counts.
        
        Returns:
            Counter object with class counts
        """
        return Counter(self.labels)
    
    def print_class_distribution(self):
        """Print class distribution."""
        class_counts = self.get_class_distribution()
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")
    
    def get_image_dimensions(self, sample_size=500):
        """
        Analyze image dimensions from a sample of images.
        
        Args:
            sample_size: Number of images to sample for analysis
            
        Returns:
            dict with 'widths' and 'heights' lists
        """
        widths = []
        heights = []
        
        sample_paths = self.image_paths[:sample_size]
        
        for path in sample_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                pass
        
        return {'widths': widths, 'heights': heights}
    
    def print_dimension_stats(self, sample_size=500):
        """Print image dimension statistics."""
        dims = self.get_image_dimensions(sample_size)
        widths = dims['widths']
        heights = dims['heights']
        
        print(f"\nSampled {len(widths)} images for dimension analysis.")
        print(f"Average width: {sum(widths)/len(widths):.1f}")
        print(f"Average height: {sum(heights)/len(heights):.1f}")
    
    def plot_sample_images(self, figsize=(15, 8)):
        """
        Plot one random image per class.
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        unique_classes = sorted(set(self.labels))
        
        fig = plt.figure(figsize=figsize)
        
        for i, cls in enumerate(unique_classes):
            # Find all image indices for this class
            cls_indices = [idx for idx, lbl in enumerate(self.labels) if lbl == cls]
            
            # Pick a random image
            rand_idx = random.choice(cls_indices)
            img_path = self.image_paths[rand_idx]
            img = Image.open(img_path)
            
            # Plot
            plt.subplot(1, len(unique_classes), i + 1)
            plt.imshow(img)
            plt.title(cls, fontsize=10)
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    def plot_mean_faces(self, pattern='*males', figsize=(18, 12)):
        """
        Plot mean faces per class/subgroup.
        
        Args:
            pattern: Glob pattern to match subgroup directories
            figsize: Figure size tuple (width, height)
            
        Note:
            Reproduced from https://github.com/visionjo/facerec-bias-bfw/blob/master/code/notebooks/1a_generate_mean_faces.ipynb
        """
        # Set style
        font = {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'font.color': 'darkred',
            'font.weight': 'normal',
            'font.size': 16,
        }
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        sns.set_style('whitegrid', font)
        
        # Find subgroup directories
        dir_subgroups = list(Path(self.data_dir).glob(pattern))
        dir_subgroups.sort()
        
        if len(dir_subgroups) == 0:
            print(f"No directories found matching pattern '{pattern}'")
            return
        
        # Create subplots
        n_rows = (len(dir_subgroups) + 1) // 2
        fig, axs = plt.subplots(n_rows, 2, figsize=figsize)
        axs = axs.flatten() if len(dir_subgroups) > 1 else [axs]
        
        mean_images = []
        
        for dir_subgroup, ax in tqdm(zip(dir_subgroups, axs), 
                                     total=len(dir_subgroups),
                                     desc="Computing mean faces"):
            # Load and resize all images in this subgroup
            images = np.array([
                resize(read(f_image), 124, 108) 
                for f_image in Path(dir_subgroup).glob("*/*.jpg")
            ])
            
            # Compute mean face
            mean_face = np.array(images).mean(0)
            
            # Plot
            ax.imshow(mean_face)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(dir_subgroup).split('/')[-1], fontsize=20)
            
            
            mean_images.append(mean_face)
        
        # Hide unused subplots
        for ax in axs[len(dir_subgroups):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return mean_images
    
    def run_full_analysis(self, sample_size=500, plot_samples=True, plot_means=True):
        """
        Run complete dataset analysis.
        
        Args:
            sample_size: Number of images to sample for dimension analysis
            plot_samples: Whether to plot sample images per class
            plot_means: Whether to plot mean faces
        """
        print("="*60)
        print("FACE DATASET ANALYSIS")
        print("="*60)
        
        # Print classes
        self.print_classes()
        
        # Print distribution
        self.print_class_distribution()
        
        # Print dimension stats
        self.print_dimension_stats(sample_size)
        
        # Plot sample images
        if plot_samples:
            print("\nPlotting sample images...")
            self.plot_sample_images()
        
        # Plot mean faces
        if plot_means:
            print("\nComputing and plotting mean faces...")
            self.plot_mean_faces()


# Convenience function for quick analysis
def analyze_face_dataset(data_dir, sample_size=500, plot_samples=True, plot_means=True):
    """
    Convenience function to quickly analyze a face dataset.
    
    Args:
        data_dir: Path to dataset root directory
        sample_size: Number of images to sample for dimension analysis
        plot_samples: Whether to plot sample images per class
        plot_means: Whether to plot mean faces
        sample_size: Number of images to sample for dimension analysis
    Returns:
        FaceDatasetAnalyzer instance
    """
    analyzer = FaceDatasetAnalyzer(data_dir)
    analyzer.run_full_analysis(sample_size, plot_samples, plot_means)
    return analyzer

