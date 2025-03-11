import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize test data to [0,1] range - required for proper model processing and visualization
x_test = x_test / 255.0  

# Build a simple neural network
print("Building model...")
model = Sequential([
    # Convert 28x28 2D images to 1D vectors of length 784 (28*28)
    Flatten(input_shape=(28, 28)),
    # Hidden layer with 128 neurons and ReLU activation for non-linearity
    Dense(128, activation='relu'),
    # Output layer with 10 neurons (one per digit) and softmax to get probabilities
    Dense(10, activation='softmax')
])
print(model.summary())

print("Compiling model...")
model.compile(
    optimizer='adam',  # Adam optimizer - adaptive learning rate method
    loss='sparse_categorical_crossentropy',  # Appropriate loss function for integer labels
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model
print("Training model...")
# Normalize training data on-the-fly and train for 5 epochs with batches of 64 images
model.fit(x_train / 255.0, y_train, epochs=5, batch_size=64)
print("Model trained.")

# Loss function for FGSM attack
print("Defining FGSM attack...")
# Same loss function as model training - needed to compute gradients for the attack
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
print(loss_object)

def fgsm_attack(image, label, epsilon):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        image: Input image to be perturbed
        label: True label of the image
        epsilon: Attack strength parameter - controls perturbation magnitude
        
    Returns:
        A tuple containing (adversarial_image, gradient, signed_grad)
    """
    # Convert inputs to TensorFlow tensors if they aren't already
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int64)
    
    # Print the input tensors
    print("\nInput image tensor:")
    print(f"Type: {type(image)}")
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print("Values:")
    print(image.numpy())  # Convert tensor to numpy array for easier viewing
    
    print("\nLabel tensor:")
    print(f"Type: {type(label)}")
    print(f"Shape: {label.shape}")
    print(f"Data type: {label.dtype}")
    print("Values:")
    print(label.numpy())
    
    with tf.GradientTape() as tape:
        # Enable gradient recording for the image tensor
        tape.watch(image)
        # Get model prediction for the image
        prediction = model(image)
        
        # Print the prediction tensor
        print("\nPrediction tensor:")
        print(f"Shape: {prediction.shape}")
        print(f"Values (probabilities for each digit):")
        print(prediction.numpy())
        
        # Calculate loss between true label and prediction
        loss = loss_object(label, prediction)
        
        # Print the loss tensor
        print("\nLoss tensor:")
        print(f"Shape: {loss.shape}")
        print(f"Value: {loss.numpy()}")
    
    # Get the gradient of the loss w.r.t. the input image
    # This shows how the image pixels affect the loss/prediction
    gradient = tape.gradient(loss, image)
    
    # Print the gradient tensor
    print("\nGradient tensor:")
    print(f"Shape: {gradient.shape}")
    print("Gradient values (showing first few values):")
    print(gradient.numpy().flatten()[:10])  # Show just first 10 values
    
    # Get the sign of the gradients to create adversarial example
    # FGSM only uses the direction (sign) of the gradient, not magnitude
    signed_grad = tf.sign(gradient)
    
    # Print the signed gradient tensor
    print("\nSigned gradient tensor:")
    print(f"Shape: {signed_grad.shape}")
    print("Values (showing a small section):")
    if signed_grad.shape[1] >= 5 and signed_grad.shape[2] >= 5:
        print(signed_grad[0, 0:5, 0:5].numpy())  # Show a 5x5 section
    
    # Add the perturbation to the image
    # Multiply by epsilon to control attack strength
    adversarial_image = image + epsilon * signed_grad
    
    # Ensure pixel values remain in valid range [0,1]
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    # Print the final adversarial image tensor
    print("\nAdversarial image tensor:")
    print(f"Shape: {adversarial_image.shape}")
    print("Values (showing a small section):")
    if adversarial_image.shape[1] >= 5 and adversarial_image.shape[2] >= 5:
        print(adversarial_image[0, 0:5, 0:5].numpy())
    
    # Return the adversarial image along with gradient and signed gradient for visualization
    return adversarial_image, gradient, signed_grad

# Test on a single image
# Select the first test image and reshape to [1,28,28] for model input
img = x_test[0:1]
# Get corresponding label
label = y_test[0:1]

# Controls the intensity of the attack - higher values create more visible perturbations
epsilon = 0.1  
# Generate adversarial version of the test image
adv_img, gradient, signed_grad = fgsm_attack(img, label, epsilon)

# Create a figure with 4 subplots
plt.figure(figsize=(12, 8))

# Plot original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img[0], cmap='gray')
plt.colorbar()

# Plot the raw gradient (may be hard to see due to small values)
plt.subplot(2, 2, 2)
plt.title('Raw Gradient (∂Loss/∂Image)')
gradient_image = gradient.numpy()[0]
plt.imshow(gradient_image, cmap='seismic')  # seismic colormap: blue for negative, red for positive
plt.colorbar()

# Plot the signed gradient (the direction of the attack)
plt.subplot(2, 2, 3)
plt.title('Sign of Gradient (Direction of Attack)')
signed_grad_image = signed_grad.numpy()[0]
plt.imshow(signed_grad_image, cmap='seismic')  # seismic colormap: blue for -1, red for +1
plt.colorbar()

# Plot the perturbation (epsilon * sign of gradient)
plt.subplot(2, 2, 4)
plt.title(f'Perturbation (ε={epsilon})')
perturbation = (epsilon * signed_grad).numpy()[0]
plt.imshow(perturbation, cmap='seismic')
plt.colorbar()

plt.tight_layout()
plt.show()

# Now plot the original vs adversarial comparison
plt.figure(figsize=(9, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img[0], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Perturbation\n(Magnified for Visibility)')
# Magnify the perturbation to make it more visible
magnified_perturbation = perturbation * 10
plt.imshow(magnified_perturbation, cmap='seismic')

plt.subplot(1, 3, 3)
plt.title('Adversarial Image')
plt.imshow(adv_img[0], cmap='gray')

plt.tight_layout()
plt.show()

# Get model's prediction on original image (returns class with highest probability)
original_pred = np.argmax(model.predict(img))
# Get model's prediction on adversarial image
adv_pred = np.argmax(model.predict(adv_img))

# Display results to show if attack was successful (predictions differ)
print(f"Original Prediction: {original_pred}")
print(f"Adversarial Prediction: {adv_pred}")
