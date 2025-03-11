import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# Build a simple CNN model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Train the model normally
model = create_model()
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

# FGSM attack
def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = image + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0.0, 1.0)
    return adversarial_image

# Generate adversarial examples
def generate_adversarial_images(model, images, labels, epsilon):
    adv_images = []
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for i in range(len(images)):
        image = tf.convert_to_tensor(images[i:i+1])  # Create a batch of one image
        label = labels[i:i+1]
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = loss_fn(label, prediction)
        
        gradient = tape.gradient(loss, image)
        adv_image = fgsm_attack(image[0], epsilon, gradient[0])
        adv_images.append(adv_image)

    return tf.stack(adv_images)

# Set epsilon for perturbation
epsilon = 0.1

# Get adversarial examples for part of the training data
adv_x_train = generate_adversarial_images(model, x_train[:1000], y_train[:1000], epsilon)
y_adv_train = y_train[:1000]

# Combine clean and adversarial data
x_combined = tf.concat([x_train, adv_x_train], axis=0)
y_combined = tf.concat([y_train, y_adv_train], axis=0)

# Retrain the model with adversarial examples
model_adv = create_model()
model_adv.fit(x_combined, y_combined, epochs=3, batch_size=64, validation_split=0.1)

# Evaluate on clean and adversarial examples
adv_x_test = generate_adversarial_images(model_adv, x_test[:1000], y_test[:1000], epsilon)

print("Clean test accuracy:")
model_adv.evaluate(x_test, y_test)

print("Adversarial test accuracy:")
model_adv.evaluate(adv_x_test, y_test[:1000])

# Visualizing some adversarial examples with predictions
plt.figure(figsize=(10, 8))

# Get predictions
clean_preds = model_adv.predict(x_test[:5])
clean_pred_labels = np.argmax(clean_preds, axis=1)

adv_preds = model_adv.predict(adv_x_test[:5])
adv_pred_labels = np.argmax(adv_preds, axis=1)

true_labels = y_test[:5]

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(tf.squeeze(x_test[i]).numpy(), cmap='gray')
    plt.title(f"True: {true_labels[i]}\nPred: {clean_pred_labels[i]}")
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(tf.squeeze(adv_x_test[i]).numpy(), cmap='gray')
    plt.title(f"True: {true_labels[i]}\nPred: {adv_pred_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print overall accuracy comparison
print("\nAccuracy Comparison:")
print("-" * 50)
print("Original model on clean test data:")
clean_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"Accuracy: {clean_acc:.4f}")

print("\nOriginal model on adversarial test data:")
orig_adv_acc = model.evaluate(adv_x_test, y_test[:1000], verbose=0)[1]
print(f"Accuracy: {orig_adv_acc:.4f}")

print("\nAdversarially trained model on clean test data:")
adv_clean_acc = model_adv.evaluate(x_test, y_test, verbose=0)[1]
print(f"Accuracy: {adv_clean_acc:.4f}")

print("\nAdversarially trained model on adversarial test data:")
adv_adv_acc = model_adv.evaluate(adv_x_test, y_test[:1000], verbose=0)[1]
print(f"Accuracy: {adv_adv_acc:.4f}")

print("\nImprovement on adversarial examples:")
print(f"{(adv_adv_acc - orig_adv_acc) * 100:.2f}% better accuracy")
