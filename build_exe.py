import PyInstaller.__main__
import os
import sys
import shutil

# Force CPU-only PyTorch mode for packaging
os.environ['PYTORCH_JIT'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define application name
app_name = "ChangeDetectionViewer"

# Clean previous builds
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists("build"):
    shutil.rmtree("build")

# Define additional data files to include
data_files = [
    ('checkpoints/last.ckpt', 'checkpoints'),  # Include model checkpoint
    ('test', 'test'),  # Include test data directory
]

# Create a list of PyInstaller arguments
pyinstaller_args = [
    'app.py',  # Your main script
    '--name=%s' % app_name,
    '--onefile',  # Create a single executable file
    '--windowed',  # Use windowed mode (no console)
    '--clean',  # Clean PyInstaller cache
    '--icon=NONE',  # You can replace NONE with path to an .ico file
    '--add-data=checkpoints/last.ckpt:checkpoints',  # Include model checkpoint
]

# Add data directories
# PyInstaller needs a separate --add-data argument for each directory/file
for src, dest in data_files:
    if os.path.isdir(src):
        # For directories, we need to add each file individually
        for root, dirs, files in os.walk(src):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the destination path relative to the base directory
                rel_path = os.path.relpath(os.path.dirname(file_path), start=os.path.dirname(src))
                dest_path = os.path.join(dest, rel_path)
                pyinstaller_args.append(f'--add-data={file_path}:{dest_path}')
    else:
        # For individual files
        pyinstaller_args.append(f'--add-data={src}:{dest}')

# Add the src directory as a module
pyinstaller_args.append('--add-data=src:src')

# Run PyInstaller
PyInstaller.__main__.run(pyinstaller_args)

print(f"\nBuild completed! Your executable should be in the 'dist' folder.")
print(f"Executable name: {app_name}.exe")