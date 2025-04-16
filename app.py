import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import lightning as lightning
import threading
import queue
import random  # For random image selection
import sys  # For system information

# Import project-specific components
from src.change_detection import ChangeDetectionTask
from src.datasets.levircd import LEVIRCDDataModule

# Import theme colors
from theme import (
    background as BACKGROUND_COLOR,
    text as TEXT_COLOR,
    surface as SURFACE_COLOR,
    card as CARD_COLOR,
    accent as ACCENT_COLOR,
    button_bg as BUTTON_BG,
    button_fg as BUTTON_FG,
    highlight as HIGHLIGHT_COLOR,
    error as ERROR_COLOR,
    success as SUCCESS_COLOR
)

# Application version
APP_VERSION = "1.0.1"

class ThemedTk(tk.Tk):
    """Custom Tk class with theme integration"""
    def __init__(self):
        super().__init__()
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background=BACKGROUND_COLOR)
        self.style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
        self.style.configure("TButton", background=BUTTON_BG, foreground=BUTTON_FG)
        self.style.configure("TEntry", fieldbackground=SURFACE_COLOR, foreground=TEXT_COLOR)
        
        # Configure ComboBox styles - fix dropdown colors
        self.style.configure("TCombobox", fieldbackground=SURFACE_COLOR, foreground=TEXT_COLOR, 
                             background=SURFACE_COLOR, arrowcolor=TEXT_COLOR)
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', SURFACE_COLOR)],
                       selectbackground=[('readonly', BUTTON_BG)],
                       selectforeground=[('readonly', BUTTON_FG)])
        
        # Fix hover/selection colors for elements
        self.style.map('TButton', 
                      background=[('active', ACCENT_COLOR)],
                      foreground=[('active', BUTTON_FG)])
        
        # Progress bar styles
        self.style.configure("Horizontal.TProgressbar", 
                            background=ACCENT_COLOR, 
                            troughcolor=SURFACE_COLOR,
                            borderwidth=0,
                            thickness=6)
        
        # StatusBar progress bar - thinner, accent color
        self.style.configure("StatusBar.Horizontal.TProgressbar", 
                           background=ACCENT_COLOR, 
                           troughcolor=BACKGROUND_COLOR,
                           borderwidth=0,
                           thickness=4)
        
        # Card style for frame groups
        self.style.configure("Card.TFrame", background=SURFACE_COLOR)
        self.style.configure("Card.TLabel", background=SURFACE_COLOR, foreground=TEXT_COLOR)
        
        # LabelFrame style for headings like "Controls"
        self.style.configure("Card.TLabelframe", background=SURFACE_COLOR)
        self.style.configure("Card.TLabelframe.Label", background=SURFACE_COLOR, foreground=TEXT_COLOR)
        
        # Rounded button style
        self.style.configure("Rounded.TButton", 
                             relief="flat", 
                             background=BUTTON_BG,
                             foreground=BUTTON_FG,
                             padding=(10, 5))
        self.style.map('Rounded.TButton', 
                      background=[('active', ACCENT_COLOR)],
                      foreground=[('active', BUTTON_FG)])
        
        # Status message styles with color coding
        self.style.configure("success.TLabel", foreground=SUCCESS_COLOR, background=SURFACE_COLOR)
        self.style.configure("error.TLabel", foreground=ERROR_COLOR, background=SURFACE_COLOR)
        self.style.configure("info.TLabel", foreground=TEXT_COLOR, background=SURFACE_COLOR)
        
        # Header label style
        self.style.configure("Header.TLabel", 
                            foreground=TEXT_COLOR, 
                            background=BACKGROUND_COLOR, 
                            font=("Arial", 16, "bold"))
        
        # Configure master window
        self.configure(background=BACKGROUND_COLOR)
        
        # Set up option menu styling for dropdowns
        self.option_add('*TCombobox*Listbox.background', SURFACE_COLOR)
        self.option_add('*TCombobox*Listbox.foreground', TEXT_COLOR)
        self.option_add('*TCombobox*Listbox.selectBackground', BUTTON_BG)
        self.option_add('*TCombobox*Listbox.selectForeground', BUTTON_FG)

class ChangeDetectionApp(ThemedTk):
    def __init__(self):
        super().__init__()
        
        self.title("Change Detection Viewer")
        self.geometry("1200x800")
        
        # Model parameters
        self.checkpoints_dir = "checkpoints"
        self.available_checkpoints = self.get_available_checkpoints()
        self.checkpoint_path = "checkpoints/last.ckpt"
        self.data_root = "./test"
        self.batch_size = 8
        self.patch_size = 256
        self.workers = 8  # Using 0 for the UI to avoid multiprocessing issues
        
        # Initialize total_images to a default value
        self.total_images = 0
        
        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Create UI elements
        self.create_ui()
        
        # Start queue processing
        self.process_queue()
        
        # Load model and dataset in a separate thread but don't process any images yet
        self.status_var.set("Loading model and dataset...")
        threading.Thread(target=self.setup_model_thread, daemon=True).start()
        
        # Register cleanup method when window is closed
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_available_checkpoints(self):
        """Get a list of available checkpoint files"""
        checkpoints = []
        if os.path.exists(self.checkpoints_dir):
            for file in os.listdir(self.checkpoints_dir):
                if file.endswith(".ckpt"):
                    checkpoints.append(os.path.join(self.checkpoints_dir, file))
        return checkpoints if checkpoints else ["checkpoints/last.ckpt"]
    
    def setup_model(self):
        """Load the model and dataset"""
        try:
            # Load dataset using the same approach as show_random_detection_result
            self.datamodule = LEVIRCDDataModule(
                root=self.data_root,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                num_workers=self.workers,
            )
            self.datamodule.setup("test")
            self.dataset = self.datamodule.test_dataloader().dataset
            self.total_images = len(self.dataset)
            
            # Load model using the same approach as show_random_detection_result
            self.model = ChangeDetectionTask.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
            print(f"Model loaded successfully. Dataset contains {self.total_images} images.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or dataset: {str(e)}")
            raise e
    
    def setup_model_thread(self):
        """Load the model and dataset in a background thread"""
        try:
            # Load dataset
            datamodule = LEVIRCDDataModule(
                root=self.data_root,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                num_workers=self.workers,
            )
            datamodule.setup("test")
            dataset = datamodule.test_dataloader().dataset
            total_images = len(dataset)
            
            # Load model
            model = ChangeDetectionTask.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
            
            # Add to queue to update UI from main thread
            self.queue.put(("model_loaded", {
                "datamodule": datamodule,
                "dataset": dataset,
                "total_images": total_images,
                "model": model
            }))
        except Exception as e:
            self.queue.put(("error", {"message": f"Failed to load model or dataset: {str(e)}"}))

    def process_queue(self):
        """Process messages from background threads"""
        try:
            while not self.queue.empty():
                message_type, data = self.queue.get(0)
                
                if message_type == "model_loaded":
                    # Update application with loaded model and dataset
                    self.datamodule = data["datamodule"]
                    self.dataset = data["dataset"]
                    self.total_images = data["total_images"]
                    self.model = data["model"]
                    
                    # Hide progress indicator
                    self.progress.stop()
                    self.progress.pack_forget()
                    
                    # Update UI to reflect loaded model
                    self.status_var.set(f"Model loaded successfully. Dataset contains {self.total_images} images.")
                    self.status_bar.configure(style="success.TLabel")
                    
                    # Update the image index range label
                    for widget in self.img_control_frame.winfo_children():
                        if isinstance(widget, ttk.Label) and "(0-" in widget.cget("text"):
                            widget.config(text=f"(0-{self.total_images-1})")
                            break
                    
                    # Don't automatically load the first image
                    # Let the user select an image explicitly
                    
                elif message_type == "image_processed":
                    # Display the processed image
                    self.display_image_results(data)
                    
                elif message_type == "error":
                    # Hide progress indicator
                    self.progress.stop()
                    self.progress.pack_forget()
                    
                    # Show error message
                    messagebox.showerror("Error", data["message"])
                    self.status_var.set("Error occurred")
                    self.status_bar.configure(style="error.TLabel")
                
        except Exception as e:
            print(f"Queue processing error: {e}")
        
        # Schedule next queue check (fixed to avoid recursion)
        self.after(100, lambda: self.process_queue())

    def create_ui(self):
        """Create the user interface components"""
        # Main frame
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # App header with branding
        header_frame = ttk.Frame(main_frame, style="TFrame", height=60)
        header_frame.pack(fill=tk.X, pady=10)
        header_frame.pack_propagate(False)  # Maintain fixed height

        # App title with larger font
        title_label = ttk.Label(header_frame, text="Change Detection Viewer", 
                            style="Header.TLabel")
        title_label.pack(side=tk.LEFT, padx=20)

        # Version info
        version_label = ttk.Label(header_frame, text=f"v{APP_VERSION}", style="TLabel")
        version_label.pack(side=tk.RIGHT, padx=20)
        
        # Top control area - use the Card style for control areas
        control_frame = ttk.LabelFrame(main_frame, text="Controls", style="Card.TLabelframe")
        control_frame.pack(fill=tk.X, pady=15, padx=5)
        
        # Model checkpoint selection
        model_frame = ttk.Frame(control_frame, style="Card.TFrame")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_frame, text="Model Checkpoint:", style="Card.TLabel").pack(side=tk.LEFT, padx=10)
        self.checkpoint_var = tk.StringVar(value=self.checkpoint_path)
        checkpoint_dropdown = ttk.Combobox(model_frame, textvariable=self.checkpoint_var, 
                                          values=self.available_checkpoints, width=40)
        checkpoint_dropdown.pack(side=tk.LEFT, padx=10)
        
        # Create a themed button for model loading with icon
        load_model_btn = ttk.Button(model_frame, text="⟲ Load Model", command=self.load_model, style="Rounded.TButton")
        load_model_btn.pack(side=tk.LEFT, padx=10)
        
        # Add image selection - use Card style for consistent appearance
        img_control_frame = ttk.Frame(control_frame, style="Card.TFrame")
        img_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Store reference to this frame for queue processing
        self.img_control_frame = img_control_frame
        
        ttk.Label(img_control_frame, text="Image Index:", style="Card.TLabel").pack(side=tk.LEFT, padx=10)
        self.index_var = tk.StringVar(value="0")
        index_entry = ttk.Entry(img_control_frame, textvariable=self.index_var, width=6)
        index_entry.pack(side=tk.LEFT, padx=5)
        
        range_label = ttk.Label(img_control_frame, 
                                text=f"(0-{self.total_images-1 if hasattr(self, 'total_images') else 0})",
                                style="Card.TLabel")
        range_label.pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons with themed appearance and icons
        load_img_btn = ttk.Button(img_control_frame, text="🔍 Load Image", 
                                command=self.load_selected_image_thread, 
                                style="Rounded.TButton")
        load_img_btn.pack(side=tk.LEFT, padx=10)
        
        prev_btn = ttk.Button(img_control_frame, text="⏮ Previous", 
                            command=self.prev_image, 
                            style="Rounded.TButton")
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(img_control_frame, text="Next ⏭", 
                            command=self.next_image, 
                            style="Rounded.TButton")
        next_btn.pack(side=tk.LEFT, padx=5)
        
        random_btn = ttk.Button(img_control_frame, text="🔀 Random", 
                              command=self.random_image, 
                              style="Rounded.TButton")
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display area with card styling
        display_label = ttk.Label(main_frame, text="Detection Results", style="Header.TLabel")
        display_label.pack(anchor=tk.W, pady=(15, 5), padx=5)
        
        # Create a card-style frame for the image display
        image_card = ttk.Frame(main_frame, style="Card.TFrame")
        image_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_frame = ttk.Frame(image_card, style="Card.TFrame", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar with themed appearance - FIXED to prevent going off screen
        status_frame = ttk.Frame(main_frame, style="Card.TFrame", height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        status_frame.pack_propagate(False)  # Maintain fixed height
        
        # Create a container for the status elements
        status_container = ttk.Frame(status_frame, style="Card.TFrame")
        status_container.pack(fill=tk.BOTH, expand=True)
        
        # Status text on left side
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(status_container, textvariable=self.status_var, 
                              anchor=tk.W, style="info.TLabel", padding=(10, 5))
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # System info on right side
        system_info = f"Python {sys.version.split()[0]} | {os.name.upper()}"
        system_label = ttk.Label(status_container, text=system_info, 
                               style="Card.TLabel", padding=(10, 5))
        system_label.pack(side=tk.RIGHT)
        
        # Progress bar for operations - now added permanently to the top of status bar
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, 
                                     mode="indeterminate", 
                                     style="StatusBar.Horizontal.TProgressbar")
        self.progress.pack(side=tk.TOP, fill=tk.X, padx=2, pady=0)
        self.progress.pack_forget()  # Hidden by default, but will appear in fixed position
        
        # Initialize current index but don't load any image
        self.current_index = 0
        
        # Add keyboard shortcuts
        self.bind("<Left>", lambda event: self.prev_image())
        self.bind("<Right>", lambda event: self.next_image())
        self.bind("<space>", lambda event: self.random_image())
        self.bind("<Return>", lambda event: self.load_selected_image_thread())
    
    def load_selected_image(self):
        """Load and display the selected image along with change detection results"""
        try:
            # Get index from entry
            idx = int(self.index_var.get())
            if idx < 0 or idx >= self.total_images:
                messagebox.showerror("Invalid Index", f"Index must be between 0 and {self.total_images-1}")
                return
            
            self.current_index = idx
            self.index_var.set(str(idx))
            
            # Clear previous plots
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            # Get data sample
            data = self.dataset[idx]
            image1, image2, target = data["image1"], data["image2"], data["mask"]
            
            # Process images using the same approach as show_random_detection_result
            image_pair = torch.cat([image1, image2], dim=0)
            image_pair = image_pair.unsqueeze(0)  # Add batch dimension
            
            prediction = self.model(image_pair)
            
            # Process prediction exactly as in show_random_detection_result
            prediction = prediction.squeeze().detach().cpu().numpy()
            prediction = (prediction > 0.5).astype(np.uint8)  # Threshold at 0.5
            
            # Convert tensors to displayable format exactly like in show_random_detection_result
            image1_display = image1.squeeze().permute(1, 2, 0).cpu().numpy()
            image1_display = (image1_display * 255).astype(np.uint8)
            image1_pil = Image.fromarray(image1_display)
            
            image2_display = image2.squeeze().permute(1, 2, 0).cpu().numpy()
            image2_display = (image2_display * 255).astype(np.uint8)
            image2_pil = Image.fromarray(image2_display)
            
            target_display = target.squeeze().cpu().numpy()
            target_display = (target_display * 255).astype(np.uint8)
            target_pil = Image.fromarray(target_display)
            
            # Process prediction (threshold AFTER extracting the channel)
            if prediction.ndim != 2:
                prediction = prediction.squeeze()
            
            # Select the change detection channel (index 1)
            prediction = prediction[1]  # Select the channel for change detection
            
            # Now apply the threshold and reshape to match image dimensions
            prediction = prediction.reshape(image1_pil.size[::-1])  # Reshape to match image dimensions
            prediction = prediction.astype(np.uint8)  # Ensure data type is uint8
            
            # Create prediction image
            prediction_image = Image.fromarray(prediction * 255)
            
            # Create color map exactly as in show_random_detection_result
            color_map = np.zeros((256, 3), dtype=np.uint8)
            color_map[255] = [255, 255, 0]  # Yellow for change (255 in the binary mask)
            color_map[0] = [128, 0, 128]    # Purple for no change (0 in the binary mask)
            
            prediction_np = np.array(prediction_image)
            prediction_colored = Image.fromarray(color_map[prediction_np])
            prediction_colored = prediction_colored.convert("RGB")
            
            # Create matplotlib figure with the same layout
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            
            # Display images as in show_random_detection_result
            axes[0].imshow(image1_pil)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            axes[1].imshow(image2_pil)
            axes[1].set_title("Changed Image")
            axes[1].axis("off")
            
            axes[2].imshow(target_pil, cmap="gray")
            axes[2].set_title("Target")
            axes[2].axis("off")
            
            axes[3].imshow(prediction_colored, cmap="gray")
            axes[3].set_title("Detection Result")
            axes[3].axis("off")
            
            # Add the plot to tkinter window
            canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update status
            self.status_var.set(f"Loaded image {idx} of {self.total_images-1}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            print(f"Error details: {e}")
    
    def next_image(self):
        """Load the next image"""
        new_index = min(self.current_index + 1, self.total_images - 1)
        self.index_var.set(str(new_index))
        self.load_selected_image_thread()

    def prev_image(self):
        """Load the previous image"""
        new_index = max(self.current_index - 1, 0)
        self.index_var.set(str(new_index))
        self.load_selected_image_thread()

    def random_image(self):
        """Load a random image"""
        import random
        new_index = random.randint(0, self.total_images - 1)
        self.index_var.set(str(new_index))
        self.load_selected_image_thread()
    
    def load_model(self):
        """Load the selected model checkpoint"""
        try:
            selected_checkpoint = self.checkpoint_var.get()
            
            # Check if file exists
            if not os.path.exists(selected_checkpoint):
                messagebox.showerror("Error", f"Checkpoint file not found: {selected_checkpoint}")
                return
            
            # Show progress indicator in the status bar
            self.progress.pack(before=self.status_bar, fill=tk.X, padx=5, pady=2)
            self.progress.start(10)
                
            self.status_var.set(f"Loading model from {selected_checkpoint}...")
            self.status_bar.configure(style="info.TLabel")
            self.update()  # Update the UI to show the status change
            
            # Start thread to load the model
            threading.Thread(
                target=self.load_model_thread,
                args=(selected_checkpoint,),
                daemon=True
            ).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start model loading: {str(e)}")
            self.status_var.set("Error loading model")
            self.status_bar.configure(style="error.TLabel")
            
            # Hide progress on error
            self.progress.stop()
            self.progress.pack_forget()

    def load_model_thread(self, selected_checkpoint):
        """Load the model in a background thread"""
        try:
            # Load the new model
            model = ChangeDetectionTask.load_from_checkpoint(selected_checkpoint, map_location="cpu")
            
            # Send results to main thread
            self.queue.put(("model_loaded", {
                "datamodule": self.datamodule,
                "dataset": self.dataset,
                "total_images": self.total_images,
                "model": model,
                "checkpoint_path": selected_checkpoint
            }))
        except Exception as e:
            self.queue.put(("error", {"message": f"Failed to load model: {str(e)}"}))

    def on_closing(self):
        """Clean up resources before closing the application"""
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear any references to large objects
        if hasattr(self, 'model'):
            self.model = None
        
        # Delete any tkinter widgets if needed
        if hasattr(self, 'image_frame'):
            for widget in self.image_frame.winfo_children():
                widget.destroy()
        
        # Make sure to destroy the window and quit the app
        self.destroy()
        
        # Force garbage collection
        import gc
        gc.collect()

    def load_selected_image_thread(self):
        """Start a thread to load and process the selected image"""
        try:
            # Get index from entry
            idx = int(self.index_var.get())
            if idx < 0 or idx >= self.total_images:
                messagebox.showerror("Invalid Index", f"Index must be between 0 and {self.total_images-1}")
                return
            
            self.current_index = idx
            self.index_var.set(str(idx))
            
            # Clear previous plots
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            # Show progress indicator in the status bar
            self.progress.pack(before=self.status_bar, fill=tk.X, padx=5, pady=2)
            self.progress.start(10)
            
            # Show loading message
            self.status_var.set(f"Processing image {idx}...")
            self.status_bar.configure(style="info.TLabel")
            self.update()
            
            # Start thread for processing
            threading.Thread(
                target=self.process_image_thread,
                args=(idx,),
                daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start image processing: {str(e)}")
            self.status_var.set("Error processing image")
            self.status_bar.configure(style="error.TLabel")
            # Hide progress bar
            self.progress.stop()
            self.progress.pack_forget()

    def process_image_thread(self, idx):
        """Process the image in a background thread"""
        try:
            # Get data sample
            data = self.dataset[idx]
            image1, image2, target = data["image1"], data["image2"], data["mask"]
            
            # Process images using the same approach as show_random_detection_result
            image_pair = torch.cat([image1, image2], dim=0)
            image_pair = image_pair.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                prediction = self.model(image_pair)
            
            # Process prediction exactly as in show_random_detection_result
            prediction = prediction.squeeze().detach().cpu().numpy()
            prediction = (prediction > 0.5).astype(np.uint8)  # Threshold at 0.5
            
            # Convert tensors to displayable format
            image1_display = image1.squeeze().permute(1, 2, 0).cpu().numpy()
            image1_display = (image1_display * 255).astype(np.uint8)
            image1_pil = Image.fromarray(image1_display)
            
            image2_display = image2.squeeze().permute(1, 2, 0).cpu().numpy()
            image2_display = (image2_display * 255).astype(np.uint8)
            image2_pil = Image.fromarray(image2_display)
            
            target_display = target.squeeze().cpu().numpy()
            target_display = (target_display * 255).astype(np.uint8)
            target_pil = Image.fromarray(target_display)
            
            # Process prediction (threshold AFTER extracting the channel)
            if prediction.ndim != 2:
                prediction = prediction.squeeze()
            
            # Select the change detection channel (index 1)
            prediction = prediction[1]  # Select the channel for change detection
            
            # Now apply the threshold and reshape to match image dimensions
            prediction = prediction.reshape(image1_pil.size[::-1])  # Reshape to match image dimensions
            prediction = prediction.astype(np.uint8)  # Ensure data type is uint8
            
            # Create prediction image
            prediction_image = Image.fromarray(prediction * 255)
            
            # Create color map exactly as in show_random_detection_result
            color_map = np.zeros((256, 3), dtype=np.uint8)
            color_map[255] = [255, 255, 0]  # Yellow for change (255 in the binary mask)
            color_map[0] = [128, 0, 128]    # Purple for no change (0 in the binary mask)
            
            prediction_np = np.array(prediction_image)
            prediction_colored = Image.fromarray(color_map[prediction_np])
            prediction_colored = prediction_colored.convert("RGB")
            
            # Send results to main thread
            self.queue.put(("image_processed", {
                "image1_pil": image1_pil,
                "image2_pil": image2_pil,
                "target_pil": target_pil,
                "prediction_colored": prediction_colored,
                "idx": idx
            }))
        except Exception as e:
            self.queue.put(("error", {"message": f"Failed to process image: {str(e)}"}))

    def display_image_results(self, data):
        """Display the processed image results in the UI (called from main thread)"""
        try:
            # Extract data
            image1_pil = data["image1_pil"]
            image2_pil = data["image2_pil"]
            target_pil = data["target_pil"]
            prediction_colored = data["prediction_colored"]
            idx = data["idx"]
            
            # Hide and stop progress indicator now that processing is done
            self.progress.stop()
            self.progress.pack_forget()
            
            # Configure matplotlib to use our dark theme before creating any figure
            plt.style.use('dark_background')
            
            # Set default figure facecolor to match our background to prevent white flash
            plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR
            plt.rcParams['axes.facecolor'] = SURFACE_COLOR
            plt.rcParams['savefig.facecolor'] = BACKGROUND_COLOR
            
            # Create the figure with proper background color from the start
            fig = plt.figure(figsize=(15, 5), facecolor=BACKGROUND_COLOR)
            fig.patch.set_facecolor(BACKGROUND_COLOR)  # Ensure background is set
            fig.subplots_adjust(wspace=0.05)
            
            # Create grid of subplots
            axes = []
            for i in range(4):
                ax = fig.add_subplot(1, 4, i+1)
                ax.set_facecolor(SURFACE_COLOR)
                axes.append(ax)
            
            # Display images
            axes[0].imshow(image1_pil)
            axes[0].set_title("Original Image", color=TEXT_COLOR)
            axes[0].axis("off")
            
            axes[1].imshow(image2_pil)
            axes[1].set_title("Changed Image", color=TEXT_COLOR)
            axes[1].axis("off")
            
            axes[2].imshow(target_pil, cmap="gray")
            axes[2].set_title("Target", color=TEXT_COLOR)
            axes[2].axis("off")
            
            axes[3].imshow(prediction_colored)
            axes[3].set_title("Detection Result", color=TEXT_COLOR)
            axes[3].axis("off")
            
            # Create a themed frame for the matplotlib canvas with a border
            canvas_frame = ttk.Frame(self.image_frame, style="Card.TFrame")
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add a subtle border
            canvas_frame.configure(borderwidth=1, relief="solid")
            
            # Configure the background of the canvas before adding it to the frame
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.get_tk_widget().configure(bg=BACKGROUND_COLOR, highlightbackground=BACKGROUND_COLOR, highlightcolor=BACKGROUND_COLOR)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add a toolbar frame with theme-consistent buttons
            toolbar_frame = ttk.Frame(self.image_frame, style="Card.TFrame")
            toolbar_frame.pack(fill=tk.X, pady=10)
            
            # Add controls for navigation or export
            export_btn = ttk.Button(
                toolbar_frame, 
                text="💾 Export Results",
                command=lambda: self.export_results(fig, prediction_colored),
                style="Rounded.TButton"
            )
            export_btn.pack(side=tk.RIGHT, padx=10, pady=5)
            
            # Add navigation buttons in the toolbar
            prev_btn = ttk.Button(
                toolbar_frame, 
                text="⏮ Previous",
                command=self.prev_image,
                style="Rounded.TButton"
            )
            prev_btn.pack(side=tk.LEFT, padx=10, pady=5)
            
            next_btn = ttk.Button(
                toolbar_frame, 
                text="Next ⏭",
                command=self.next_image,
                style="Rounded.TButton"
            )
            next_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Update status with success message and themed color
            self.status_var.set(f"✓ Loaded image {idx} of {self.total_images-1}")
            self.status_bar.configure(style="success.TLabel")
            
        except Exception as e:
            # Hide progress indicator in case of error
            self.progress.stop()
            self.progress.pack_forget()
            
            # Show error message
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")
            print(f"Display error details: {e}")
            
            # Update status with error message
            self.status_var.set(f"Error displaying results: {str(e)}")
            self.status_bar.configure(style="error.TLabel")
    
    def export_results(self, fig, prediction_colored):
        """Export the results to image files"""
        try:
            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Results"
            )
            
            if save_path:
                # Save the figure
                fig.savefig(save_path, facecolor=BACKGROUND_COLOR, dpi=300)
                
                # Save the prediction separately
                pred_path = save_path.replace(".png", "_prediction.png")
                prediction_colored.save(pred_path)
                
                self.status_var.set(f"Results saved to {os.path.basename(save_path)}")
                self.status_bar.configure(style="success.TLabel")
                messagebox.showinfo("Export Successful", 
                                   f"Results exported to:\n{save_path}\n{pred_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            self.status_var.set(f"Error exporting results: {str(e)}")
            self.status_bar.configure(style="error.TLabel")

if __name__ == "__main__":
    app = ChangeDetectionApp()
    app.mainloop()