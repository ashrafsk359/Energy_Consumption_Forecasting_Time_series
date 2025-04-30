import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define paths
visualization_folder = "outputs/visualizations"
output_path = "outputs/model_comparison.png"

# Performance metrics
metrics = {
    "ARIMA": {"MAE": 0.1034, "RMSE": 0.1388, "MAPE": "14.66%"},
    "LSTM": {"MAE": 0.0513, "RMSE": 0.0710, "MAPE": "7.99%"},
    "XGBoost": {"MAE": 0.0486, "RMSE": 0.0687, "MAPE": "7.67%"},
}

# Load images
model_names = ["arima", "lstm", "xgb"]
images = []
for model in model_names:
    img_path = os.path.join(visualization_folder, f"{model}.png")
    if os.path.exists(img_path):
        images.append(mpimg.imread(img_path))
    else:
        images.append(None)

# Plot the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, model in enumerate(model_names):
    if images[i] is not None:
        axes[i].imshow(images[i])
        model_key = "XGBoost" if model == "xgb" else model.upper()
        axes[i].set_title(f"{model_key}\nMAE: {metrics[model_key]['MAE']}\nRMSE: {metrics[model_key]['RMSE']}\nMAPE: {metrics[model_key]['MAPE']}")

    axes[i].axis("off")

plt.suptitle("Model Comparison: ARIMA vs LSTM vs XGBoost", fontsize=14)
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Model comparison image saved at {output_path}")
