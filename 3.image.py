import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image


def main():
    st.title('Image Web Classifier')    
    st.write('Upload any image within the classes')
    

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        resized_image = image.resize((32,32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1,32,32,3))

        model = tf.keras.models.load_model('3.model.h5')

        predictions = model.predict(img_array)
        name_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse','Ship', 'Truck']

        st.success('This image is most likely: ' + name_classes[np.argmax(predictions)])
    else:
        st.text('You have not upload an image yet.')








if __name__== '__main__':
    main()