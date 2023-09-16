import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('garbage_classification_model.h5')

image_size = (150, 150)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


frame_width = 150
frame_height = 150

# Set the frame width and height for the camera capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a window to display the camera feed
window_name = "Camera Feed"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to the desired width and height
    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)

    # Display the prediction on the frame
    label = class_labels[predicted_class]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame in the window
    cv2.imshow(window_name, frame)

    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy the window when done
cap.release()
cv2.destroyAllWindows()