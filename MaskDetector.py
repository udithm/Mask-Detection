from keras.models import load_model
import cv2
import numpy as np

def DetectMask(image):
    model = load_model('model-020.model')
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(100,100))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,100,100,1))
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    
    if label == 0: return True
    else: return False
