import cv2
import numpy as np
from keras.models import load_model

model = load_model('garbage_classification_model.h5')

cap = cv2.VideoCapture(1)  

while True:
    ret, frame = cap.read()  

    if not ret:
        break

    # Preprocess the frame (resize to match the model's input size and normalize)
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0 
    
    prediction = model.predict(frame.reshape(1, 150, 150, 3))
    predicted_class = np.argmax(prediction)
    class_labels = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
    label = class_labels[predicted_class]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
