import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def main():
    # ============================
    # 1. Configuration and Paths
    # ============================

    # Define the paths to the dataset directories
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Define the model classes
    model_classes = ['earphones', 'home_appliances', 'laptop', 'phone']
    num_classes = len(model_classes)

    # Directory to save the trained model
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'smartphone_model_classifier_with_brands.keras')  # Changed extension to .keras

    # ==============================
    # 2. Data Preparation and Aug
    # ==============================

    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data should not be augmented
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load and preprocess the training images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the validation images
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Load and preprocess the test images
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # ============================
    # 3. Building the Model
    # ============================

    # Load the MobileNetV2 base model with pre-trained ImageNet weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model initially

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # ============================
    # 4. Compiling the Model
    # ============================

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    # ============================
    # 5. Training the Model
    # ============================

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_save_dir, 'best_model.keras'),  # Changed extension to .keras
                                           monitor='val_loss',
                                           save_best_only=True)
    ]

    # Train the model
    initial_epochs = 20
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ==================================
    # 6. Fine-Tuning the Model
    # ==================================

    # Unfreeze the base model for fine-tuning
    base_model.trainable = True

    # Optionally, freeze some layers to prevent overfitting
    # For example, freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Re-compile the model with a lower learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=['accuracy']
    )

    # Fine-tune the model
    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks,
        verbose=1
    )

    # ==================================
    # 7. Evaluating the Model
    # ==================================

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'\nTest accuracy: {test_acc:.2f}')

    # Save the fine-tuned model
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    # ==================================
    # 8. Plotting Training History
    # ==================================

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy (Initial)')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Initial)')
    plt.plot(history_fine.history['accuracy'], label='Train Accuracy (Fine-tuned)')
    plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy (Fine-tuned)')
    plt.title('Model Accuracy During Training and Fine-Tuning')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(model_save_dir, 'accuracy_plot.png'))
    plt.show()

    # ==================================
    # 9. Confusion Matrix and Report
    # ==================================

    # Predict on the test set
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # Actual labels
    y_true = test_generator.classes

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=model_classes,
                yticklabels=model_classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix for Smartphone Models')
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix.png'))
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=model_classes)
    print('Classification Report:\n', report)

    # Optionally, save the classification report to a text file
    with open(os.path.join(model_save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()
