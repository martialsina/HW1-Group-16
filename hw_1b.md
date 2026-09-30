## Hw 1b: Experimental Analysis of a Shallow Neural Network

Neural networks are powerful function approximators, but their performance depends heavily on the choice of architecture and training parameters. In this part of the assignment, you will systematically explore how different hyperparameters and training strategies impact the performance of a shallow neural network trained on the Fashion-MNIST dataset. These experiments will help you develop an intuition for model tuning and performance tradeoffs in real-world settings.

We will use the **Adam** optimizer for all experiments. The dataset contains **grayscale 28×28 images** of clothing items, and your task is to classify each image into one of **10 categories**. You are encouraged to reuse and extend your code from **Part 1a**.

---

### 0. Complete `training_utils.py`

Before starting the experiments below, complete the following functions in `training_utils.py`:

- `get_model`: returns a freshly initialized 2-layer neural network.
- `train_model`: trains a model for a given number of epochs and returns the final training loss.
- `valid_metrics`: evaluates a trained model on a validation set and returns the validation loss and accuracy.

All experiments in this assignment should call these functions rather than duplicating training/evaluation logic in the notebook.

---

### 1. Pick a Validation Set

Use the **first 10,000 examples** from the training set as your **validation set**. The remaining examples will be used for training.

---

### 2. Learning Rate Selection

Train a 2-layer neural network with the following configuration:

- Hidden layer size: `50`
- Activation: ReLU
- Optimizer: Adam
- Epochs: 500

Train separate models using the following learning rates:

`[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]`


For each setting, **reinitialize the model from scratch** (i.e., do not reuse weights).

Then, **interpolate between the best two values** by testing additional learning rates (e.g., 0.005, 0.002, etc.) to identify the optimal learning rate.

Train each model for 500 epochs and report the training loss, validation loss, and validation accuracy as a function of learning rate in a table.


---

### 3. Effect of Hidden Layer Size

Fix the learning rate to the best value found above. Now investigate how the size of the hidden layer affects model performance.

Train models with the following hidden sizes:

`[10, 50, 100, 300, 1000, 2000]`

Train each model for 500 epochs and report the training loss, validation loss, and validation accuracy as a function of hidden size in a table.

---
### 4. Visualize transformed images

Pick a set of transformations (e.g., random crop, flip, moderate brightness/contrast) and visualize four images along with their transformed versions.

---

### 5. Effect of Data Augmentation

In this section, you will explore how **data augmentation** affects model performance using the functions you implemented earlier.

Use the **best learning rate** and **best hidden size** from your previous experiments.

Design a set of **augmentation pipelines**, each applying a different combination and intensity of augmentations. For example:

- **Baseline**: no augmentations
- **Mild**: random crop, horizontal flip
- **Moderate**: random crop, flip, moderate brightness/contrast
- **Aggressive**: random crop, flip, stronger brightness/contrast, random rotation

You may use or modify the `apply_augmentations` function from `training_utils.py`. All augmentation levels crop-and-pad back to the original 28×28 size, so training and validation images stay the same shape throughout. Do **not** apply augmentation to validation or test images — evaluate them as-is.

Train a model for each augmentation level and report the training loss, validation loss, and validation accuracy as a function of transformation level in a table.

---

## Deliverables

- Use `training_utils.py` to define any reusable code, such as:
  - A function that trains a model and logs results
  - A function that returns a freshly initialized model
  - Any augmentation functions

- A **notebook** that runs all experiments and presents tables and plots. The notebook should include **minimal inline code**: most logic should be imported from `training_utils.py`.
