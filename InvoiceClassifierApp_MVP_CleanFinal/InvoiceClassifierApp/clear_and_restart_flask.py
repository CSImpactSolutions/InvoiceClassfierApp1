
import os
import shutil
import subprocess

# Step 1: Remove __pycache__ folders
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name == "__pycache__":
            full_path = os.path.join(root, name)
            print(f"Removing cache folder: {full_path}")
            shutil.rmtree(full_path)

# Step 2: Remove .pyc files
for root, dirs, files in os.walk(".", topdown=True):
    for file in files:
        if file.endswith(".pyc"):
            full_path = os.path.join(root, file)
            print(f"Deleting bytecode: {full_path}")
            os.remove(full_path)

# Step 3: Restart Flask app
print("\n✅ Cache cleared. Restarting Flask UI...")
subprocess.run(["python", "invoice_ui_app.py"])
