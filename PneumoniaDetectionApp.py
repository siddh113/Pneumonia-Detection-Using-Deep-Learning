import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit.web import cli as stcli
from streamlit import runtime
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

IMAGE_SIZE = 224
MODEL_PATH_1 = 'D:\Vgg16_model.h5'
MODEL_PATH_2 = 'D:\CNN_Model1.h5'
MODEL_PATH_3 = 'D:\ResNet50V2.h5'

model_1 = load_model(MODEL_PATH_1)
model_2 = load_model(MODEL_PATH_2)
model_3 = load_model(MODEL_PATH_3)

def main():
    session_state = st.session_state.setdefault('results', {'df': pd.DataFrame(columns=['Image', 'Model', 'Result', 'Precision', 'Recall', 'F1-score'])})

    st.title('Pneumonia Detection Using chest X-rays')

    model_option = st.selectbox('Select a model to use:', ['VGG16', 'CNN model', 'ResNet50V2'])

    if model_option == 'VGG16':
        model = model_1
    elif model_option =='CNN model':
        model = model_2
    else:
        model = model_3

    uploaded_files = st.file_uploader('Drag and drop Normal or Pneumonia images', accept_multiple_files=True)
    counter=0
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(244, 244))

            st.image(image, caption='Uploaded Image', use_column_width=True)
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            if st.button(f'Predict {uploaded_file.name}', key=uploaded_file.name):
                placeholder = st.empty()
                with st.spinner('Predicting...'):
                    prediction = model.predict(img_array)
                if prediction[0][0] > 0.5:
                    result = 'Pneumonia'
                else:
                    result = 'Normal'
                precision, recall, f1_score, _ = classification_report([result], [result], output_dict=True)[result].values()
                placeholder.empty()
                st.write('Result for ' + uploaded_file.name + ' using the ' + model_option + ' model is: ' + result)
                session_state['df'] = session_state['df'].append({'Image': uploaded_file.name, 'Model': model_option, 'Result': result, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}, ignore_index=True)

        if st.button('Save Predictions'):
            session_state['saved_results'] = session_state['df']
        
        if st.button('Save results to .csv'):
            session_state['df'].to_csv('predictions.csv', index=False)

        if 'saved_results' in session_state:
            st.write('Saved Predictions:')
            st.dataframe(session_state['saved_results'])

    else:
        if 'saved_results' in session_state:
            st.write('Saved Predictions:')
            st.dataframe(session_state['saved_results'])

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
