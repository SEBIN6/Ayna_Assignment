# Polygon Colorization with UNet — Ayna ML Internship Assignment

## 🧠 Objective

The primary objective is to train a UNet model from scratch that can accurately generate a colored image of a polygon based on two inputs: a grayscale image of the polygon and the name of a target color.

## 📦 Dataset

The dataset is organized in a `dataset` folder containing `inputs` (grayscale polygons), `outputs` (the corresponding colored polygons), and a `data.json` file. The `data.json` file links each input image to its ground truth output and the specified color name. A custom PyTorch `Dataset` class, `PolygonColorDataset`, was created to handle this specific data format.

## 🏗️ Model Architecture: UNet with Conditional Input

The model is a UNet, an encoder-decoder architecture well-suited for image-to-image translation tasks. It was implemented in PyTorch from scratch.

### Conditioning Strategy
To incorporate the color information, the model's input is a 4-channel tensor. The 1-channel grayscale input image is concatenated with a 3-channel color map. This color map is created by expanding the 3-element RGB color vector to match the spatial dimensions of the input image, providing a consistent color signal to the model.

---

## ⚙️ Hyperparameters

| Hyperparameter | Rationale | Final Setting |
| :--- | :--- | :--- |
| **Learning Rate** | A standard starting point for the Adam optimizer. | `1e-3` |
| **Optimizer** | The Adam optimizer is a robust choice for a wide range of deep learning tasks. | `torch.optim.Adam` |
| **Loss Function** | Mean Squared Error (MSE) is an appropriate choice for this image regression task, as the model predicts pixel values. | `nn.MSELoss()` |
| **Epochs** | A sufficient number of epochs to allow the model to converge without overfitting. | `10` |
| **Batch Size** | A standard batch size that balances memory usage and training stability. | `16` |
| **Image Resolution** | A low resolution was chosen to speed up training and inference, which is sufficient for simple polygon shapes. | `128x128` |

## 📊 Training Dynamics

Training and validation losses were tracked using [wandb](https://www.wandb.ai).

### Loss Curves
The loss curves show a consistent and healthy decrease in both training and validation loss over 10 epochs. [cite_start]The training loss decreased from approximately `0.1095` to `0.0249` [cite: 1, 2][cite_start], while the validation loss dropped from `0.1086` to `0.0243`[cite: 1, 2].

### Sample Outputs
The model successfully learned to fill polygons with the specified colors, as seen in the qualitative outputs. The predictions became cleaner and more accurate as the training progressed.

| Input Polygon | Target Color | Model Output |
|---------------|--------------|--------------|
| ![gray](inputs/triangle.png) | `"blue"` | ![output](outputs/triangle_blue.png) |

### Failure Modes
The primary potential failure mode identified is the model's possible struggle with imperfect or noisy input images, which could lead to a less precise distinction between the polygon and the background.

---

## 📚 Key Learnings

* **Multi-modal Input**: An effective way to condition a model on non-image data (like a color name) is to represent it as an expanded channel in the input tensor.
* **UNet Effectiveness**: The UNet architecture is highly effective for image-to-image translation tasks, leveraging skip connections to preserve spatial details.
* **Experiment Tracking**: Tools like Weights & Biases are crucial for visualizing training dynamics and confirming that the model is learning and generalizing well without overfitting.

---

## 🚀 Deliverables

* `Unet_Model.ipynb`
* `inference.ipynb`
* `unet_model.pth`
* `README.md` (this file)
* [cite_start][wandb Project Link](https://wandb.ai/sebin2308-mbccet/ayna-polygon-coloring) [cite: 1]
* [cite_start][wandb Run Link](https://wandb.ai/sebin2308-mbccet/ayna-polygon-coloring/runs/lszf1isu) [cite: 2]
