from tensorflow.keras.applications.densenet import *
from tensorflow.keras.preprocessing import image
import numpy as np

model = DenseNet201(weights='imagenet')

! rm *.jpg
! wget https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg
! mv *.jpg image.jpg

img = image.load_img('image.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
