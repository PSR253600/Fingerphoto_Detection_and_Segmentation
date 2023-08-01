from ultralytics import YOLO
import streamlit as st
import PIL
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import base64
from io import BytesIO


#Absolute path of the current py file
file_path = Path(__file__).resolve()


#Parent directory of the current file
root_path = file_path.parent


#Adding the root path to the sys path list if not present
if root_path not in sys.path:
    sys.path.append(str(root_path))


#Relative path of the root directory in accordance with the current working directory
root = root_path.relative_to(Path.cwd())


model_dir=root/'weights' #Relative path of the weights directory
detection_model=model_dir/'best_detected.pt' #YOLOv8 Pre-trained Detection Model
segmentation_model=model_dir/'best_segmented.pt' #YOLOv8 Pre-trained Segmentation Model

#To crop fingerprint regions from the image
def extract_regions_from_boxes(image, boxes):
    """
    Extract regions from the input image based on the given bounding boxes.

    Parameters:
        image (PIL.Image): The input image.
        boxes (List[List[float]]): List of bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        List[PIL.Image]: List of extracted regions as PIL.Image objects.
    """
    regions = []
    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = map(int, box)
        region = image.crop((x_min, y_min, x_max, y_max))
        regions.append(region)
    return regions


#Execution of the detection model with the uploaded images
def process_detected_img(conf, model):
    src_imgs = st.sidebar.file_uploader("Choose image(s):", type={"png", "jpeg", "jpg", "bmp", "tif"}, accept_multiple_files=True) #File uploader
    if not src_imgs:
        st.sidebar.warning('Please upload image(s)')
        return

    uploaded_images = [] #List to store the uploaded images

    for ids, src_img in enumerate(src_imgs):
        try:
            uploaded_image = PIL.Image.open(src_img)
            uploaded_images.append(uploaded_image)
            # st.image(uploaded_image, caption=f"Uploaded Image {ids+1}", use_column_width=True)
        except Exception as ex:
            st.error(f"Error occurred while opening the image {ids+1}.") #File may have corrupted
            st.error(ex)
    
    st.write("Press the button to detect the fingerprint(s) from the image(s).")

    if st.button('Detect Fingerprints'):
        for ids, uploaded_image in enumerate(uploaded_images):
            col1, col2 = st.columns(2)

            with col1:
                #Display the uploaded image
                st.image(uploaded_image, caption=f"Uploaded Image {ids+1}", use_column_width=True)

            with col2:
                #Current date and time as a string
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                #Original file name and extension
                original_name, extension = os.path.splitext(src_imgs[ids].name)

                #Unique file name for every outcome using timestamp
                file_name = f"{original_name}_{timestamp}{extension}"

                res = model.predict(uploaded_image, conf=conf) #Prediction using pre-trained model
                boxes = res[0].boxes #Region of fingerprints in an image
                if len(boxes) > 0:
                    detected_regions = extract_regions_from_boxes(uploaded_image, boxes)

                    for i, region in enumerate(detected_regions):
                        # Display the extracted region
                        st.image(region, caption=f"Detected Region {i + 1}", use_column_width=True)

                        #Unique file name for every outcome using timestamp
                        region_file_name = f"{original_name}_region_{i+1}{extension}"
                        
                        # Create a temporary directory to store extracted region files
                        temp_dir = "temp_regions"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_filename = os.path.join(temp_dir, region_file_name)
                        with open(temp_filename, "wb") as f:
                            region.save(f, format='PNG')

                        # Read the saved file as bytes
                        with open(temp_filename, "rb") as f:
                            region_bytes = f.read()

                        # Encode the binary data to Base64
                        b64_data = base64.b64encode(region_bytes).decode()

                        # Provide the download link for the extracted image
                        region_download_link = f'<a href="data:file/png;base64,{b64_data}" ' \
                                               f'download="{region_file_name}">Click to download Region {i + 1}</a>'
                        st.markdown(region_download_link, unsafe_allow_html=True)

                        # Remove the temporary file
                        os.remove(temp_filename)

                
                res_plotted = res[0].plot()[:, :, ::-1] #Required detection/segmentation

                #Converting the NumPy array to a PIL image before conversion
                res_plotted_pil = PIL.Image.fromarray(np.uint8(res_plotted))

                # Convert to RGB if needed
                # if res_plotted_pil.mode != 'RGB':
                #     res_plotted_pil = res_plotted_pil.convert('RGB')

                #Display the resulting image
                st.image(res_plotted_pil, caption=f'Resulting Image {ids+1}', use_column_width=True)

                #BytesIO object to store the processed image
                temp_buffer = BytesIO()
                res_plotted_pil.save(temp_buffer, format='PNG')

                #Encoding the binary data to Base64
                base64_data = base64.b64encode(temp_buffer.getvalue()).decode()

                #Download link for each image
                download_link = f'<a href="data:file/png;base64,{base64_data}" download="{file_name}">Click to download Resulting Image {ids+1}</a>'
                st.markdown(download_link, unsafe_allow_html=True)

                # Close the BytesIO buffer
                temp_buffer.close()


#Execution of the segmentation model with the uploaded images
def process_segmented_img(conf, model):
    src_imgs = st.sidebar.file_uploader("Choose image(s):", type={"png", "jpeg", "jpg", "bmp", "tif"}, accept_multiple_files=True) #File uploader
    if not src_imgs:
        st.sidebar.warning('Please upload image(s)')
        return

    uploaded_images = [] #List to store the uploaded images

    for ids, src_img in enumerate(src_imgs):
        try:
            uploaded_image = PIL.Image.open(src_img)
            uploaded_images.append(uploaded_image)
            # st.image(uploaded_image, caption=f"Uploaded Image {ids+1}", use_column_width=True)
        except Exception as ex:
            st.error(f"Error occurred while opening the image {ids+1}.") #File may have corrupted
            st.error(ex)
    
    st.write("Press the button to segment the fingerprint(s) from the image(s).")

    if st.button('Segment Fingerprints'):
        for ids, uploaded_image in enumerate(uploaded_images):
            col1, col2 = st.columns(2)

            with col1:
                #Display the uploaded image
                st.image(uploaded_image, caption=f"Uploaded Image {ids+1}", use_column_width=True)

            with col2:
                #Current date and time as a string
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                #Original file name and extension
                original_name, extension = os.path.splitext(src_imgs[ids].name)

                #Unique file name for every outcome using timestamp
                file_name = f"{original_name}_{timestamp}{extension}"

                res = model.predict(uploaded_image, conf=conf) #Prediction using pre-trained model
                boxes = res[0].boxes #Region of fingerprints in an image
                res_plotted = res[0].plot()[:, :, ::-1] #Required detection/segmentation

                #Converting the NumPy array to a PIL image before conversion
                res_plotted_pil = PIL.Image.fromarray(np.uint8(res_plotted))

                # Convert to RGB if needed
                # if res_plotted_pil.mode != 'RGB':
                #     res_plotted_pil = res_plotted_pil.convert('RGB')

                if len(boxes) > 0:
                    detected_regions = extract_regions_from_boxes(res_plotted_pil, boxes)

                    for i, region in enumerate(detected_regions):
                        # Display the extracted region
                        st.image(region, caption=f"Detected Region {i + 1}", use_column_width=True)

                        #Unique file name for every outcome using timestamp
                        region_file_name = f"{original_name}_region_{i+1}{extension}"
                        
                        # Create a temporary directory to store extracted region files
                        temp_dir = "temp_regions"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_filename = os.path.join(temp_dir, region_file_name)
                        with open(temp_filename, "wb") as f:
                            region.save(f, format='PNG')

                        # Read the saved file as bytes
                        with open(temp_filename, "rb") as f:
                            region_bytes = f.read()

                        # Encode the binary data to Base64
                        b64_data = base64.b64encode(region_bytes).decode()

                        # Provide the download link for the extracted image
                        region_download_link = f'<a href="data:file/png;base64,{b64_data}" ' \
                                               f'download="{region_file_name}">Click to download Region {i + 1}</a>'
                        st.markdown(region_download_link, unsafe_allow_html=True)

                        # Remove the temporary file
                        os.remove(temp_filename)

                #Display the resulting image
                st.image(res_plotted_pil, caption=f'Resulting Image {ids+1}', use_column_width=True)

                #BytesIO object to store the processed image
                temp_buffer = BytesIO()
                res_plotted_pil.save(temp_buffer, format='PNG')

                #Encoding the binary data to Base64
                base64_data = base64.b64encode(temp_buffer.getvalue()).decode()

                #Download link for each image
                download_link = f'<a href="data:file/png;base64,{base64_data}" download="{file_name}">Click to download Resulting Image {ids+1}</a>'
                st.markdown(download_link, unsafe_allow_html=True)

                # Close the BytesIO buffer
                temp_buffer.close()


#Page Specifications
st.set_page_config(
    page_title="Fingerprint Detection and Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Title of the page
st.title("Fingerprint Detection and Segmentation (YOLOv8)")

#Title of the sidebar
st.sidebar.header("YOLOv8 Custom Model Configuration")

model_type=st.sidebar.radio("Select Task",['Detection','Segmentation'])

confidence=st.sidebar.slider('Model Confidence Threshold',min_value=0.0,max_value=1.0,value=0.3)

if model_type=='Detection':
    model_path=detection_model
elif model_type=='Segmentation':
    model_path=segmentation_model

#Loading the pre-trained YOLOv8 model
try:
    model=YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the path: {model_path}") #Model may be missing
    st.error(ex)

if model_type=='Detection':
    process_detected_img(confidence,model)
elif model_type=='Segmentation':
    process_segmented_img(confidence,model)