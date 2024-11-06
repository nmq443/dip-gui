import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.title('Vision GUI')

st.header('Abnormal detection using subtraction')

def bgr_to_rgb(img):
    rgb = Image.fromarray(img[..., ::-1])
    return rgb

st.subheader('Input')
col1, col2 = st.columns(2)
with col1:
    error_img = st.file_uploader(
        label='Error image',
        key='error_img'
    )
with col2: 
    original_img = st.file_uploader(
        label='Original image',
        key='original_img'
    ) 

if error_img and original_img:
    # PIL image
    error_img_uploaded = Image.open(error_img)
    original_img_uploaded = Image.open(original_img)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image=error_img_uploaded, caption='Uploaded image')
    with col2:
        st.image(image=original_img, caption='Original image')
    subtracted = cv2.subtract(np.array(original_img_uploaded), np.array(error_img_uploaded))
    st.image(image=subtracted, caption='After subtraction')

st.header("Template matching")
st.subheader('Input')
col1, col2 = st.columns(2)

with col1:
    img_upload = st.file_uploader(
        label='Image',
        key='img'
    )
with col2:
    template_upload = st.file_uploader(
        label='Object/Template',
        key='object'
    )


if img_upload and template_upload:
    def template_matching():
        img_uploaded = Image.open(img_upload)
        template_uploaded = Image.open(template_upload)

        img_uploaded = np.array(img_uploaded)
        template_uploaded = np.array(template_uploaded)

        img_uploaded = cv2.cvtColor(img_uploaded, cv2.COLOR_BGR2GRAY)
        template_uploaded = cv2.cvtColor(template_uploaded, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img_uploaded, template_uploaded, cv2.TM_CCOEFF_NORMED)

        w, h = img_uploaded.shape[0], img_uploaded.shape[1]

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img_uploaded, top_left, bottom_right, 255, 2)
        st.image(img_uploaded)
    template_matching()

st.header("Filters")
st.subheader('Input')
col1, col2 = st.columns(2)
with col1:
    img_upload = st.file_uploader(
        label='Image',
        key='non_filtered_img'
    )
with col2:
    filter = st.radio(
        label='Filter',
        options=['Histogram Equalization', 'Adaptive Histogram Equalization', 'Constrast-Limited Adaptive Histogram Equalization'],
        key='filter selection'
    )

if img_upload and filter:
    img_uploaded = Image.open(img_upload)
    img_uploaded = np.array(img_uploaded)
    gray_img = cv2.cvtColor(img_uploaded, cv2.COLOR_BGR2GRAY)

    if filter == 'Histogram Equalization':
        filtered_image = cv2.equalizeHist(gray_img)

    elif filter == "Adaptive Histogram Equalization":
        tile_grid_size = (8, 8)  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid_size)
        filtered_image = clahe.apply(gray_img)

    elif filter == "Contrast-Limited Adaptive Histogram Equalization (CLAHE)":
        tile_grid_size = (8, 8)  
        clip_limit = 2.0  
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        filtered_image = clahe.apply(gray_img)

    st.image(image=filtered_image, caption='Filtered image')
