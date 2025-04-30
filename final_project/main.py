import subprocess
import os
import shutil

# Define paths
models_folder = "models"
output_folder = "outputs"
visuals_folder = os.path.join(output_folder, "visualizations")

# Ensure output and visualization folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(visuals_folder, exist_ok=True)

# Define model scripts
models = ["arima.py", "lstm.py", "xgb.py"]

# Run each model and save output
for model in models:
    model_name = model.split(".")[0]
    model_path = os.path.join(models_folder, model)
    output_path = os.path.join(output_folder, f"{model_name}_output.txt")
    
    with open(output_path, "w") as output_file:
        process = subprocess.Popen(["python", model_path], stdout=output_file, stderr=subprocess.STDOUT)
        process.wait()

    # Move generated visualizations to the output folder
    for file in os.listdir(models_folder):
        if file.startswith(model_name) and file.endswith(".png"):
            shutil.move(os.path.join(models_folder, file), os.path.join(visuals_folder, file))

print("All models have been run. Outputs and visualizations are saved in the 'outputs' folder.")
