import os
import sys
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from src.change_detection import ChangeDetectionTask
    from src.datasets.levircd import LEVIRCDDataModule
except ImportError:
    messagebox.showerror(
        "Error",
        "Could not import required modules. Make sure the src directory exists.",
    )
    sys.exit(1)


class ChangeDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Change Detection Tool")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")

        self.image1_path = None
        self.image2_path = None
        self.label_path = None
        self.checkpoint_path = "checkpoints/last.ckpt"
        self.model = None
        self.threshold = 0.5  # Default threshold value for prediction

        # Data folder selection
        self.data_folder = "train"  # Default to train folder
        
        # Data paths
        self.update_data_paths()

        # LEVIRCDDataModule parameters
        self.patch_size = 256
        self.batch_size = 1
        self.workers = 2
        self.datamodule = None

        self.create_widgets()

        self.load_model()

        # Register cleanup method when window is closed
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def update_data_paths(self):
        """Update data paths based on selected folder"""
        # Set data paths based on selected folder (train or test)
        self.data_dir = self.data_folder
        self.data_a_dir = os.path.join(self.data_dir, "A")
        self.data_b_dir = os.path.join(self.data_dir, "B")
        self.data_label_dir = os.path.join(self.data_dir, "label")
        self.data_root = f"./{self.data_folder}"  # For LEVIRCDDataModule

    def create_widgets(self):
        top_frame = tk.Frame(self, bg="#f0f0f0", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text="Checkpoint:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.checkpoint_entry = tk.Entry(top_frame, width=40)
        self.checkpoint_entry.insert(0, self.checkpoint_path)
        self.checkpoint_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Browse", command=self.browse_checkpoint).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(top_frame, text="Load Model", command=self.load_model).pack(
            side=tk.LEFT, padx=5
        )
        
        # Data folder selection frame
        folder_frame = tk.Frame(self, bg="#f0f0f0", pady=10)
        folder_frame.pack(fill=tk.X)
        
        folder_label = tk.Label(
            folder_frame, 
            text="Select Data Folder:", 
            bg="#f0f0f0",
            font=("Arial", 10, "bold")
        )
        folder_label.pack(side=tk.LEFT, padx=5)
        
        # Radio buttons for folder selection
        self.folder_var = tk.StringVar(value=self.data_folder)
        
        train_radio = tk.Radiobutton(
            folder_frame,
            text="Train Dataset",
            variable=self.folder_var,
            value="train",
            bg="#f0f0f0",
            command=self.on_folder_changed
        )
        train_radio.pack(side=tk.LEFT, padx=10)
        
        test_radio = tk.Radiobutton(
            folder_frame,
            text="Test Dataset",
            variable=self.folder_var,
            value="test",
            bg="#f0f0f0",
            command=self.on_folder_changed
        )
        test_radio.pack(side=tk.LEFT, padx=10)

        train_frame = tk.Frame(self, bg="#f0f0f0", pady=10)
        train_frame.pack(fill=tk.X)

        # Update the label to indicate we're selecting from the current data folder
        self.image_selection_label = tk.Label(
            train_frame,
            text=f"Select {self.data_folder} Image:",
            bg="#f0f0f0",
            font=("Arial", 10, "bold"),
        )
        self.image_selection_label.pack(side=tk.LEFT, padx=5)

        self.train_images = self.get_dataset_images()
        self.train_image_var = tk.StringVar()
        self.train_image_combo = ttk.Combobox(
            train_frame, textvariable=self.train_image_var, width=50
        )
        self.train_image_combo["values"] = self.train_images
        self.train_image_combo.pack(side=tk.LEFT, padx=5)
        self.train_image_combo.bind(
            "<<ComboboxSelected>>", self.on_train_image_selected
        )

        # Button to load selected training image
        tk.Button(
            train_frame,
            text="Load Selected Images",
            command=self.load_selected_training_images,
            bg="#3498db",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)

        # Middle frame for image selection
        middle_frame = tk.Frame(self, bg="#f0f0f0", pady=10)
        middle_frame.pack(fill=tk.X)

        # Image 1 selection
        img1_frame = tk.Frame(middle_frame, bg="#f0f0f0")
        img1_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Label(img1_frame, text="Image 1 (Before):", bg="#f0f0f0").pack(anchor=tk.W)
        self.image1_entry = tk.Entry(img1_frame, width=50)
        self.image1_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(img1_frame, text="Browse", command=lambda: self.browse_image(1)).pack(
            side=tk.LEFT, padx=5
        )

        # Image 2 selection
        img2_frame = tk.Frame(middle_frame, bg="#f0f0f0")
        img2_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Label(img2_frame, text="Image 2 (After):", bg="#f0f0f0").pack(anchor=tk.W)
        self.image2_entry = tk.Entry(img2_frame, width=50)
        self.image2_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(img2_frame, text="Browse", command=lambda: self.browse_image(2)).pack(
            side=tk.LEFT, padx=5
        )

        # Process button
        process_frame = tk.Frame(self, bg="#f0f0f0", pady=10)
        process_frame.pack(fill=tk.X)

        # Add threshold slider
        threshold_frame = tk.Frame(process_frame, bg="#f0f0f0", padx=10, pady=5)
        threshold_frame.pack(fill=tk.X)

        tk.Label(
            threshold_frame,
            text="Detection Threshold:",
            bg="#f0f0f0",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        self.threshold_value = tk.DoubleVar(value=self.threshold)
        self.threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            orient="horizontal",
            resolution=0.05,
            length=300,
            variable=self.threshold_value,
            command=self.update_threshold,
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=10)

        self.threshold_label = tk.Label(
            threshold_frame, text=f"Value: {self.threshold:.2f}", bg="#f0f0f0", width=10
        )
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        # Process button
        buttons_frame = tk.Frame(process_frame, bg="#f0f0f0", pady=10)
        buttons_frame.pack()

        self.process_button = tk.Button(
            buttons_frame,
            text="Detect Changes",
            command=self.process_images,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
        )
        self.process_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(
            process_frame, orient=tk.HORIZONTAL, length=400, mode="determinate"
        )
        self.progress.pack(pady=5)

        # Results frame
        self.results_frame = tk.Frame(self)
        self.results_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_checkpoint(self):
        filepath = filedialog.askopenfilename(
            title="Select Checkpoint File",
            filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")],
        )
        if filepath:
            self.checkpoint_path = filepath
            self.checkpoint_entry.delete(0, tk.END)
            self.checkpoint_entry.insert(0, filepath)

    def browse_image(self, image_num):
        filepath = filedialog.askopenfilename(
            title=f"Select Image {image_num}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            if image_num == 1:
                self.image1_path = filepath
                self.image1_entry.delete(0, tk.END)
                self.image1_entry.insert(0, filepath)
            else:
                self.image2_path = filepath
                self.image2_entry.delete(0, tk.END)
                self.image2_entry.insert(0, filepath)

    def load_model(self):
        try:
            self.checkpoint_path = self.checkpoint_entry.get()
            if not os.path.exists(self.checkpoint_path):
                messagebox.showerror(
                    "Error", f"Checkpoint file not found: {self.checkpoint_path}"
                )
                return

            self.status_var.set("Loading model...")
            self.update()

            # Load model from checkpoint
            self.model = ChangeDetectionTask.load_from_checkpoint(
                self.checkpoint_path, map_location="cpu"
            )

            # Force the model type to be "unet" regardless of what's in the checkpoint
            # This ensures consistency with how we process images
            if self.model.hparams["model"] != "unet":
                self.status_var.set(
                    f"Converting {self.model.hparams['model']} to unet model..."
                )
                self.model.hparams["model"] = "unet"

            self.status_var.set("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed!")

    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to 256x256 which is the expected patch_size
            img = img.resize((256, 256))

            # Convert to tensor
            img_np = np.array(img)

            # Apply exact same normalization as in LEVIR-CD datamodule's test_aug
            # First normalize from [0,255] to [0,1]
            img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()
            img_tensor = img_tensor / 255.0
            
            # Then normalize from [0,1] to [-1,1] as in levircd.py
            img_tensor = (img_tensor - 0.5) / 0.5

            return img_tensor
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            return None

    def process_images(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return

        # Get image paths from entries
        self.image1_path = self.image1_entry.get()
        self.image2_path = self.image2_entry.get()

        if not self.image1_path or not self.image2_path:
            messagebox.showerror("Error", "Please select both before and after images!")
            return

        if not os.path.exists(self.image1_path) or not os.path.exists(self.image2_path):
            messagebox.showerror("Error", "One or both image files do not exist!")
            return

        try:
            self.status_var.set("Processing images...")
            self.progress["value"] = 20
            self.update()

            # Preprocess images
            img1_tensor = self.preprocess_image(self.image1_path)
            self.progress["value"] = 40
            self.update()

            img2_tensor = self.preprocess_image(self.image2_path)
            self.progress["value"] = 60
            self.update()

            if img1_tensor is None or img2_tensor is None:
                return

            # Match the processing approach in app.py
            # Concatenate images along channel dimension and add batch dimension
            image_pair = torch.cat([img1_tensor, img2_tensor], dim=0)
            image_pair = image_pair.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                prediction = self.model(image_pair)

            self.progress["value"] = 80
            self.update()

            # Process prediction exactly as in app.py
            prediction = prediction.squeeze().detach().cpu().numpy()
            
            # Process prediction (threshold AFTER extracting the channel)
            if prediction.ndim != 2:
                prediction = prediction.squeeze()
            
            # Select the change detection channel (index 1)
            prediction_prob = prediction[1]  # Select the channel for change detection
            
            # Now apply the threshold and reshape to match image dimensions
            prediction_prob = prediction_prob.reshape(256, 256)  # Match patch_size dimensions
            prediction_binary = (prediction_prob > self.threshold).astype(np.uint8)

            # Store the last processed tensors and prediction for threshold updates
            self.last_img1_tensor = img1_tensor.detach()
            self.last_img2_tensor = img2_tensor.detach()
            self.last_prediction = prediction_prob

            # Display results
            self.display_results(img1_tensor, img2_tensor, prediction_binary)

            self.progress["value"] = 100
            self.status_var.set(
                f"Change detection completed! (threshold={self.threshold:.2f})"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error during change detection: {str(e)}")
            self.status_var.set(f"Change detection failed! Error: {str(e)}")
            import traceback

            traceback.print_exc()

    def process_images_with_levircd(self):
        """Process images using the LEVIRCDDataModule for exact same preprocessing as in notebook"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return

        # Get image paths from entries
        self.image1_path = self.image1_entry.get()
        self.image2_path = self.image2_entry.get()

        if not self.image1_path or not self.image2_path:
            messagebox.showerror("Error", "Please select both before and after images!")
            return

        if not os.path.exists(self.image1_path) or not os.path.exists(self.image2_path):
            messagebox.showerror("Error", "One or both image files do not exist!")
            return

        try:
            self.status_var.set("Processing images using LEVIR-CD module...")
            self.progress["value"] = 20
            self.update()

            # Initialize the datamodule
            self.datamodule = LEVIRCDDataModule(
                root=self.data_root,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                num_workers=self.workers,
            )

            # Load and preprocess images using PIL
            img1 = Image.open(self.image1_path).convert("RGB")
            img2 = Image.open(self.image2_path).convert("RGB")

            # Resize to patch size
            img1 = img1.resize((self.patch_size, self.patch_size))
            img2 = img2.resize((self.patch_size, self.patch_size))

            # Convert to numpy arrays
            img1_np = np.array(img1)
            img2_np = np.array(img2)

            # Convert to tensors (C, H, W)
            img1_tensor = torch.from_numpy(img1_np.transpose((2, 0, 1))).float()
            img2_tensor = torch.from_numpy(img2_np.transpose((2, 0, 1))).float()

            # Apply the same normalization as in LEVIR-CD datamodule's test_aug
            # First normalize from [0,255] to [0,1]
            img1_tensor = img1_tensor / 255.0
            img2_tensor = img2_tensor / 255.0

            # Then normalize from [0,1] to [-1,1]
            img1_tensor = (img1_tensor - 0.5) / 0.5
            img2_tensor = (img2_tensor - 0.5) / 0.5

            self.progress["value"] = 60
            self.update()

            # Run the model with preprocessed tensors
            model_type = self.model.hparams["model"]

            with torch.no_grad():
                if model_type == "unet":
                    # For UNet models, concatenate along channel dimension
                    x = torch.cat([img1_tensor, img2_tensor], dim=0)
                    x = x.unsqueeze(0)  # Add batch dimension
                    prediction = self.model(x)
                elif model_type in ["fcsiamdiff", "fcsiamconc"]:
                    # For Siamese models, stack them with an extra dimension
                    x = torch.stack((img1_tensor, img2_tensor), dim=0)
                    x = x.unsqueeze(0)  # Add batch dimension
                    prediction = self.model(x)
                elif model_type in ["bit", "changeformer", "tinycd"]:
                    # These models take two separate images
                    img1_tensor = img1_tensor.unsqueeze(0)  # Add batch dimension
                    img2_tensor = img2_tensor.unsqueeze(0)  # Add batch dimension
                    prediction = self.model(img1_tensor, img2_tensor)
                else:
                    # Fallback for unknown models
                    x = torch.cat([img1_tensor, img2_tensor], dim=0)
                    x = x.unsqueeze(0)
                    prediction = self.model(x)

            self.progress["value"] = 80
            self.update()

            # Process prediction
            prediction = prediction.squeeze().detach().cpu().numpy()

            # Debug information
            self.status_var.set(
                f"Prediction shape: {prediction.shape}, min: {prediction.min():.4f}, max: {prediction.max():.4f}"
            )

            # Get change probability (typically channel index 1 for binary classification with 2 classes)
            if len(prediction.shape) > 2:
                if prediction.shape[0] == 2:  # Binary classification with 2 channels
                    prediction_prob = prediction[
                        1
                    ]  # Get channel 1 (change probability)
                else:
                    prediction_prob = prediction[0]  # Default to first channel
            else:
                prediction_prob = prediction

            # Normalize the prediction to 0-1 range if outside that range
            if prediction_prob.min() < 0 or prediction_prob.max() > 1:
                prediction_prob = (prediction_prob - prediction_prob.min()) / (
                    prediction_prob.max() - prediction_prob.min()
                )
                self.status_var.set(
                    f"Normalized prediction: min: {prediction_prob.min():.4f}, max: {prediction_prob.max():.4f}"
                )

            # Store the last processed tensors and prediction for threshold updates
            self.last_img1_tensor = img1_tensor.detach()
            self.last_img2_tensor = img2_tensor.detach()
            self.last_prediction = prediction_prob

            # Apply thresholding to get binary prediction using current threshold
            prediction_binary = (prediction_prob > self.threshold).astype(np.uint8)

            # Display results
            self.display_results(img1_tensor, img2_tensor, prediction_binary)

            self.progress["value"] = 100
            self.status_var.set(
                f"LEVIR-CD processing completed! (threshold={self.threshold:.2f})"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error during LEVIR-CD processing: {str(e)}")
            self.status_var.set(f"Processing failed! Error: {str(e)}")
            import traceback

            traceback.print_exc()

    def display_results(self, img1_tensor, img2_tensor, prediction):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Convert tensors to PIL images for display
        def tensor_to_pil(tensor):
            # Denormalize from [-1, 1] to [0, 1] to [0, 255]
            img = tensor.permute(1, 2, 0).cpu().numpy()
            # First denormalize from [-1, 1] to [0, 1]
            img = (img * 0.5) + 0.5
            # Then convert to uint8 range [0, 255]
            img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)

        img1 = tensor_to_pil(img1_tensor)
        img2 = tensor_to_pil(img2_tensor)

        # Create a binary prediction image
        prediction_image = Image.fromarray(prediction * 255)

        # Create colored prediction image - using the exact same color scheme as the notebook
        color_map = np.zeros((256, 3), dtype=np.uint8)
        color_map[0] = [128, 0, 128]  # Purple for no change
        color_map[255] = [255, 255, 0]  # Yellow for detected changes

        prediction_np = np.array(prediction_image)
        colored_prediction = Image.fromarray(color_map[prediction_np])
        colored_prediction = colored_prediction.convert("RGB")

        # Create figure for display
        fig = plt.Figure(figsize=(15, 5))

        # Add images to figure
        ax1 = fig.add_subplot(141)
        ax1.imshow(img1)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(142)
        ax2.imshow(img2)
        ax2.set_title("Changed Image")
        ax2.axis("off")

        ax3 = fig.add_subplot(143)
        # If label image exists, display it
        if self.label_path and os.path.exists(self.label_path):
            try:
                label_img = Image.open(self.label_path)
                if label_img.mode != "L":
                    label_img = label_img.convert("L")
                ax3.imshow(label_img, cmap="gray")
                ax3.set_title("Target")
            except Exception:
                ax3.imshow(prediction_image, cmap="gray")
                ax3.set_title("Target (Failed to load)")
        else:
            ax3.imshow(prediction_image, cmap="gray")
            ax3.set_title(f"Binary Prediction (t={self.threshold:.2f})")
        ax3.axis("off")

        ax4 = fig.add_subplot(144)
        ax4.imshow(colored_prediction)
        ax4.set_title("Detection Result")
        ax4.axis("off")

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add save button
        save_frame = tk.Frame(self.results_frame)
        save_frame.pack(pady=10)

        tk.Button(
            save_frame,
            text="Save Results",
            command=lambda: self.save_results(fig, colored_prediction),
        ).pack()

    def save_results(self, fig, prediction_image):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if save_path:
            try:
                fig.savefig(save_path)
                prediction_path = save_path.replace(".png", "_prediction.png")
                prediction_image.save(prediction_path)
                messagebox.showinfo(
                    "Success", f"Results saved to {save_path} and {prediction_path}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def get_training_images(self):
        """Get all training images from folder A"""
        if not os.path.exists(self.train_a_dir):
            return []

        # List all png files in the training A directory
        image_files = [f for f in os.listdir(self.train_a_dir) if f.endswith(".png")]
        image_files.sort()  # Sort alphabetically
        return image_files

    def get_dataset_images(self):
        """Get all images from the selected folder's A directory"""
        if not os.path.exists(self.data_a_dir):
            return []

        # List all png files in the selected folder's A directory
        image_files = [f for f in os.listdir(self.data_a_dir) if f.endswith(".png")]
        image_files.sort()  # Sort alphabetically
        return image_files

    def on_train_image_selected(self, event=None):
        """Callback when a training image is selected from the dropdown"""
        # Get selected image filename
        selected = self.train_image_var.get()
        if not selected:
            return

        self.status_var.set(f"Selected training image: {selected}")

    def load_selected_training_images(self):
        """Load the selected images from folders A, B, and label"""
        selected = self.train_image_var.get()
        if not selected:
            messagebox.showerror("Error", "Please select an image first!")
            return

        # Extract the base filename pattern from the selected image
        base_name = selected

        # Generate paths for corresponding images in folders A, B and label
        image_a_path = os.path.join(self.data_a_dir, base_name)
        image_b_path = os.path.join(self.data_b_dir, base_name)
        label_path = os.path.join(self.data_label_dir, base_name)

        # Check if files exist
        if not os.path.exists(image_a_path):
            messagebox.showerror(
                "Error", f"Image not found in folder A: {image_a_path}"
            )
            return

        if not os.path.exists(image_b_path):
            messagebox.showerror(
                "Error", f"Corresponding image not found in folder B: {image_b_path}"
            )
            return

        # Label image is optional for prediction
        self.label_path = label_path if os.path.exists(label_path) else None

        # Update entry fields
        self.image1_path = image_a_path
        self.image1_entry.delete(0, tk.END)
        self.image1_entry.insert(0, image_a_path)

        self.image2_path = image_b_path
        self.image2_entry.delete(0, tk.END)
        self.image2_entry.insert(0, image_b_path)

        self.status_var.set(f"Loaded images from {self.data_folder} dataset: {base_name}")

        # If both images are loaded, we could process them automatically
        if self.model is not None:
            self.process_images()

    def update_threshold(self, value):
        """Update the threshold value and update the label"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"Value: {self.threshold:.2f}")

        # If we have processed images before, re-run with the new threshold
        if hasattr(self, "last_prediction") and self.last_prediction is not None:
            # Apply thresholding to get binary prediction with new threshold
            prediction_binary = (self.last_prediction > self.threshold).astype(np.uint8)

            # Display results with new threshold
            self.display_results(
                self.last_img1_tensor, self.last_img2_tensor, prediction_binary
            )

            self.status_var.set(f"Updated threshold to {self.threshold:.2f}")

    def on_folder_changed(self):
        """Handle when the user changes between train and test folders"""
        new_folder = self.folder_var.get()
        if (new_folder != self.data_folder):
            self.data_folder = new_folder
            self.update_data_paths()
            
            # Update image dropdown with files from the selected folder
            self.train_images = self.get_dataset_images()
            self.train_image_combo["values"] = self.train_images
            
            # Clear the current selection
            self.train_image_var.set("")
            
            # Update label text based on selected dataset
            if self.data_folder == "train":
                self.status_var.set("Switched to train dataset")
            else:
                self.status_var.set("Switched to test dataset")

    def on_closing(self):
        """Clean up resources properly before closing the application"""
        # Close all matplotlib figures
        plt.close("all")

        # Clear any references to large objects
        if hasattr(self, "model"):
            self.model = None

        # Clear any stored tensors or large data
        if hasattr(self, "last_prediction"):
            self.last_prediction = None
        if hasattr(self, "last_img1_tensor"):
            self.last_img1_tensor = None
        if hasattr(self, "last_img2_tensor"):
            self.last_img2_tensor = None

        # Delete any tkinter widgets if needed
        if hasattr(self, "results_frame"):
            for widget in self.results_frame.winfo_children():
                widget.destroy()

        # Make sure to destroy the window and quit the app
        self.destroy()

        # Force garbage collection
        import gc

        gc.collect()


if __name__ == "__main__":
    app = ChangeDetectionApp()
    app.mainloop()
