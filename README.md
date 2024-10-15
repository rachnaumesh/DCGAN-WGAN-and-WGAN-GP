# GANs and WGANs on Fashion MNIST

This repository contains the implementation of DCGAN, WGAN (weight clipping), and WGAN-GP (gradient penalty) based on the paper "Improved Training of Wasserstein GANs" by Gulrajani et al. The models are trained on the Fashion MNIST dataset, and the training is supervised by the class label.

## Introduction
This project implements DCGAN, WGAN, and WGAN-GP for generating images from the Fashion MNIST dataset. Two architectures from Table 1 of the paper are used. The implementation includes:
- Training the models with class labels.
- Plotting the loss functions for each method.
- Generating images for specific labels and comparing them with real images.

## Data Preparation
The Fashion MNIST dataset is loaded and transformed to tensors using the `torchvision` library. The dataset is split into training and test sets, and the data is normalized.

## Model Architectures
Two architectures are implemented:
1. **Generator1 and Discriminator1**: Basic architecture with convolutional and deconvolutional layers.
2. **Generator2 and Discriminator2**: Enhanced architecture with additional layers and scaled shifted softplus activation.

## Training

**Set random seed for reproducibility**:
    The code sets a fixed random seed to ensure consistent results:
    ```python
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    ```
    
The models are trained using the following methods:
1. **DCGAN**: Uses Binary Cross-Entropy Loss.
2. **WGAN**: Uses Wasserstein loss with weight clipping.
3. **WGAN-GP**: Uses Wasserstein loss with gradient penalty.

### Training Steps
1. **Run the training block in notebook**
2. **Training Parameters**:
    - Number of epochs: 10
    - Batch size: 50
    - Learning rate: 0.0002 for DCGAN, 0.00005 for WGAN, 0.0001 for WGAN-GP
    - Optimizers: Adam for DCGAN and WGAN-GP, RMSprop for WGAN

## Results
### Loss Plots
The loss functions for each method are plotted and saved in the `plots` folder. Each plot contains three loss curves (one for each mode).

### Generated Images
For each architecture, generated images from each mode for four labels (e.g., dresses, sneakers, ankle boots, and trousers) are saved in the `samples` folder. Additionally, two real images from the corresponding label are included for comparison.

## Generating New Images
To generate new images using the trained models:
1. **Load the trained weights**:
    ```python
    G.load_state_dict(torch.load(f'weights/{mode}_generator_arch_{arch}.pth', weights_only=True))
    ```
2. **Generate images**:
    ```python
    noise = torch.randn(num_images, noise_dim, device=device)
    class_labels = torch.randint(0, num_classes, (num_images,))
    one_hot_labels = torch.nn.functional.one_hot(class_labels, num_classes=num_classes).float()
    generated_images = G(noise, one_hot_labels)
    save_image(generated_images, 'samples/new_generated_images.png', nrow=8, normalize=True)
    ```

## Conclusion
This project demonstrates the implementation of DCGAN, WGAN, and WGAN-GP on the Fashion MNIST dataset. The results show the effectiveness of each method in generating realistic images. The repository includes the code, trained weights, loss plots, and generated images for easy reproduction and further experimentation.

## References
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs. arXiv preprint arXiv:1704.00028.
