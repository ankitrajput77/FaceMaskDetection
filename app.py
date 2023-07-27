import streamlit as st
from src import utils
from src.components import detect_from_video
from src.components import detect_from_image
from PIL import Image
import io
st.sidebar.title('Face Mask Detector')


if 'first_run' not in st.session_state:
    st.session_state.first_run = True
else:
    st.session_state.first_run = False

if st.session_state.first_run:
    st.title("Hey!")
    st.title("This is Homepage for this app. To know more about it and how to use it, click on :blue[About This App] button.")
    st.session_state.first_run = False

if st.sidebar.button('About This App'):
    st.title('About :book:')
    st.markdown('The Face Mask Detection project is an application of machine learning and computer vision technologies that are aimed to identifying whether persons are wearing face masks or not. This project specially useful for the times when face masks are recommended or mandated to prevent the spread of diseases, such as COVID-19.')
    st.markdown('The main goal of the Face Mask Detection project is make a system that can automatically detect and differentiate faces into two categories: With Mask and Without Mask. The system uses a combination of deep learning models and image processing techniques.')
    st.title('How To Use :sunflower:')
    st.header('Web Cam')
    st.markdown('To use this project using webcam. You can just click on the Start webcam button which is located in the sidebar (under the Face Mask Detection Using WebCam text). when you click on it, it will open a webcam in the new tab and then you can click on that newtab and use your webcam for face mask detection, also to stop the webcam you can use ESC button on your keyboard which will stop your webcam and then you can use other features of this app.')
    st.header('Upload Image')
    st.markdown('To use this project using Upload Image. You can just click on the Browse files button which is located in the sidebar (under the Face Mask Detection Using Upload Image text). This Browse files button takes you to your local computer where you can select the particular image that in which you want to use detect the face masks. when you select the image this will automatically show you your image with masks and no masks. Also some statistics are also there under the image that is produced, you can check that too. ')

# About button
if st.sidebar.button('About Creator'):
    utils.about_me()

st.sidebar.header('Face Mask Detection Using Upload Image')
# file uploader 
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['jpg'])
if uploaded_file is not None:
    file = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(file))
    locs, preds = detect_from_image.from_image(img)
    img_saved_path = detect_from_image.draw_rectangle(io.BytesIO(file), locs, preds)
    img_path = img_saved_path.replace("\\", "/")
    print(img_path)
    st.image(img_path, caption="Image has been saved to the local computer in savedImages directory", use_column_width=True)

    masks = 0
    nomasks = 0
    for i in range(len(preds)):
        if preds[i][0] < preds[i][1]:
            nomasks = nomasks + 1
        else :
            masks = masks + 1
    st.info("Number of faces detected : {}".format(len(preds)))
    st.warning("Faces with mask : {}".format(masks))
    st.success("Faces without mask : {}".format(nomasks))

st.sidebar.header('Face Mask Detection Using WebCam')
if st.sidebar.button('Start WebCam'):
    
    st.title("WebCam starting in some time, usually it takes 4-5 seconds")
    detect_from_video.real_time()
st.sidebar.text("Enter ESC button to stop")
