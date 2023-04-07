import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_decision(image_path, csv_path, model_path, from_xrv):
    """
    Takes:
    image_path (str): An image path.
    model: A pytorch model (should be loaded separately before passing to this function).
    from_xrv (bool): Whether the model is from torchxrayvision.
    use_gpu (bool): Whether to use the gpu for prediction.

    Does a forward prediction pass on the image and 
    visualizes the gradient at the last layer that contributed 
    to the prediction.

    Returns the actual score, and the predicted score.
    """
    # Load model from save
    model = torch.load(model_path)

    # Load image

    # If it's from torchxrayvision, the model expects a grayscaled image
    if from_xrv:
        image = torchvision.io.read_image(image_path, mode = torchvision.io.ImageReadMode.GRAY)
        # Normalize image to what torchxrayvision expects
        image = (image - 127.5) / 127.5 * 1024 # Scales the image to between approx -1024 and 1024, from 0-255
    else:
        image = torchvision.io.read_image(image_path, mode = torchvision.io.ImageReadMode.RGB)

    # Load the ground truth label
    metadata = pd.read_csv(csv_path, index_col=0).reset_index() # Loads the csv
    image_name = os.path.basename(os.path.normpath(image_path)) # Gets the image name
    label = metadata['OpacityScoreGlobal'].loc[metadata['filename'] == image_name].values[0] # Gets the label with the correct filename

    # Process the image

    # transform image by doing a center crop to 224 pixels
    transform = transforms.Compose([
      transforms.CenterCrop(224),
      ])
    image = transform(image)

    # Set image to require grad so we can get the autograd
    image = image.requires_grad_()

    # If we have access to GPU, move image and model to GPU
    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    # Set to eval mode for reproducibility
    model.eval()

    # Make prediction on image using model
    prediction = model(image.unsqueeze(0)) # add a batch size of 1 in front so model accepts it

    # Get gradients
    gradients = torch.autograd.grad(prediction, image)[0][0]

    # Blur image of the gradients
    blurred = skimage.filters.gaussian(gradients.detach().cpu().numpy()**2, 
                                    sigma = (5,5),
                                    truncate = 3.5)

    fig, ax = plt.subplot_mosaic([['xray','preds']], figsize = (7,3.5))
    # Plot the original image on the left
    ax['xray'].imshow(image.squeeze().detach().cpu().numpy(), cmap='gray', aspect = 'auto')
    ax['xray'].set_title('Original')
    ax['xray'].axis('off')
    # Plot the original image, and the gradient overlay, on the right
    ax['preds'].imshow(image.squeeze().detach().cpu().numpy(), cmap='gray', aspect = 'auto')
    ax['preds'].imshow(blurred, alpha = 0.5)
    ax['preds'].set_title('Model focus')
    ax['preds'].axis('off')
    plt.tight_layout() # ensures that both images are the same size

    # Descriptive over-all title
    plt.suptitle(f'Actual: {label}, Prediction: {round(prediction.item(),3)}', y = 1.05);
    plt.savefig(f'{image_name}_{model_name}_pred.png')

    return label, round(prediction.item(),3)

### To use: ###
# Image and metadata stuff
# image_path = "path_to_image" # Fill this in yourself
# csv_path = "path_to_csv" # Fill this in yourself

# # Model stuff
# save_path = 'place_model_was_saved' # Fill this in yourself
# save_name = f'model_name.pt' # Fill this in yourself
# model_path = os.path.join(save_path, save_name)

# visualize_decision(image_path, csv_path, model_path, from_xrv)