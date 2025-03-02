#@title Plot training Results
from utils.save_test_visualize import load_experiment, visualize_images, visualize_attention
config, model, train_losses, test_losses, accuracies = load_experiment(f"vit-with-10-epochs/")

import matplotlib.pyplot as plt
# Create two subplots of train/test losses and accuracies
def visualize():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    plt.savefig("metrics.png")
    plt.show()


visualize_images()
visualize_attention(model, "attention.png")
