import subprocess
import streamlit as st

def run_predict(args):
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    return True

def save_uploaded_image(uploaded_file):
    # Save the uploaded image
    file_name = uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.title('YoloV5 Object Detection')
# project = st.text_input('Project Name', value='Test')
# name = st.text_input('Folder Name (Change folder name for every prediction)', value='test')
uploaded_file = st.file_uploader("Upload an image...", type="jpg")
col1, col2 = st.columns([1, 1])
if uploaded_file is not None:
    if st.button('Predict'):
        file_name = uploaded_file.name
        col1.image(uploaded_file, caption="Uploaded Image")
        save_uploaded_image(uploaded_file)
        with st.spinner('Predicting...'):
            st.text('Predicting...')
            run_predict(['python', 'predict.py', '--weights', 'best.pt', '--conf', '0.25', '--source', file_name])
            st.text('Prediction Done!')
        col2.image("predict_"+file_name, caption=f"Predicted Image")
