import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


def main():

    st.title('Digit Web Classifier')
    file = st.file_uploader('Upload a picture of numeric digit', type=['jpg', 'png'])

    if file:
        image = Image.open(file)
        st.image(image, use_column_width= True)

        image_resize = image.resize((28,28))
        image_array = np.array(image_resize) / 255
        image_array = image_array.reshape((-1,28,28,1))

        model = tf.keras.models.load_model('5.model.h5')

        prediction = model.predict(image_array)
        classes = ['0','1','2','3','4','5','6','7','8','9']

        st.success('The number uploaded is: ' + classes[np.argmax(prediction[0])])

    else:
        st.text('You have not upload an image yet.')





if __name__=='__main__':
    main()

