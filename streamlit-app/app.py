import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tempfile
import yaml
import random
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Car Detection App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CFG:
    CLASSES = ['car', 'truck', 'bus', 'motorcycle']
    NUM_CLASSES_TO_TRAIN = len(CLASSES)
    
    BASE_MODEL = 'yolov8n'  
    CONFIDENCE_THRESHOLD = 0.3
    
    IMG_SIZE = (640, 640)

@st.cache_resource(show_spinner=False)
def install_packages():
    with st.spinner("Installing required packages..."):
        os.system("pip install -q torch==2.0.1 torchvision==0.15.2")
        os.system("pip install -q ultralytics")
    
@st.cache_resource(show_spinner=True)
def load_model(model_name):
    try:
        from ultralytics import YOLO
        import torch
        
        st.sidebar.info(f"PyTorch version: {torch.__version__}")
        
        from huggingface_hub import hf_hub_download
        import os
        
        model_path = f"{model_name}.pt"
        if not os.path.exists(model_path):
            st.info(f"Downloading {model_name} model...")            
            model_repos = {
                'yolov8n': 'keremberke/yolov8n-model',
                'yolov8s': 'keremberke/yolov8s-model',
                'yolov8m': 'keremberke/yolov8m-model',
                'yolov8l': 'keremberke/yolov8l-model',
                'yolov8x': 'keremberke/yolov8x-model'
            }
            
            if model_name in model_repos:
                try:
                    model_path = hf_hub_download(
                        repo_id=model_repos[model_name],
                        filename="model.pt",
                        local_dir="./"
                    )
                    
                    if os.path.exists(model_path) and not os.path.exists(f"{model_name}.pt"):
                        os.rename(model_path, f"{model_name}.pt")
                        model_path = f"{model_name}.pt"
                except:
                    st.warning("Could not download from Hugging Face. Falling back to direct YOLO download.")
                    model_path = f"{model_name}.pt"     
            else:
                model_path = f"{model_name}.pt"     
        
        return YOLO(model_path)
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please try using a different model or check your internet connection.")
        return None

def process_image(uploaded_file, model, confidence_threshold):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        img = Image.open(BytesIO(uploaded_file.getvalue()))
        
        results = model.predict(
            source=tmp_path,
            conf=confidence_threshold,
            save=False
        )
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.array(img))
        
        boxes = results[0].boxes
        
        model_classes = model.names
        st.sidebar.write("Model classes:", model_classes)
        
        class_counts = {cls: 0 for cls in CFG.CLASSES}

        coco_to_vehicle = {
            2: "car",      
            5: "bus",      
            7: "truck",    
            3: "motorcycle" 
        }
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in model_classes:
                model_cls_name = model_classes[cls_id]
                
                if cls_id in coco_to_vehicle:
                    cls_name = coco_to_vehicle[cls_id]
                elif model_cls_name.lower() in [c.lower() for c in CFG.CLASSES]:
                    cls_name = next((c for c in CFG.CLASSES if c.lower() == model_cls_name.lower()), "unknown")
                else:
                    if model_cls_name.lower() in ["car", "automobile", "vehicle"]:
                        cls_name = "car"
                    elif model_cls_name.lower() in ["truck", "lorry"]:
                        cls_name = "truck"
                    elif model_cls_name.lower() in ["bus", "minibus"]:
                        cls_name = "bus"
                    elif model_cls_name.lower() in ["motorcycle", "motorbike"]:
                        cls_name = "motorcycle"
                    else:
                        continue
                
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                    
                    colors = {
                        "car": "blue",
                        "truck": "red",
                        "bus": "green",
                        "motorcycle": "purple"
                    }
                    color = colors.get(cls_name, "orange")
                    colors = {
                        "car": "blue",
                        "truck": "red",
                        "bus": "green",
                        "motorcycle": "purple"
                    }
                    color = colors.get(cls_name, "orange")
                    
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                        edgecolor=color, linewidth=2)
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x1, y1-10, f"{cls_name}: {conf:.2f}", 
                            color='white', fontsize=12, 
                            bbox=dict(facecolor=color, alpha=0.5))
        
        ax.axis('off')
        
        return fig, class_counts, len(boxes)
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

st.title("🚗 Car Detection Application")

st.write("""
## Detect vehicles in images using YOLO
Upload an image to detect cars, trucks, buses, and motorcycles.
""")

with st.spinner("Setting up the environment..."):
    install_packages()

with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection
    model_options = {
        'YOLOv8n': 'yolov8n',
        'YOLOv8s': 'yolov8s',
        'YOLOv8m': 'yolov8m',
        'YOLOv8l': 'yolov8l',
        'YOLOv8x': 'yolov8x'
    }
    
    selected_model_name = st.selectbox(
        'Select YOLO model:',
        list(model_options.keys())
    )
    
    model_name = model_options[selected_model_name]
    
    confidence = st.slider(
        'Confidence threshold:',
        min_value=0.1,
        max_value=1.0,
        value=CFG.CONFIDENCE_THRESHOLD,
        step=0.05
    )
    
    st.markdown("---")
    
    st.subheader("Detection Classes")
    for cls in CFG.CLASSES:
        st.write(f"- {cls.capitalize()}")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.subheader("Original Image")
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)

if 'model' not in st.session_state or st.session_state.model_name != model_name:
    with st.spinner(f"Loading {selected_model_name} model..."):
        model = load_model(model_name)
        if model:
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.sidebar.success(f"{selected_model_name} loaded successfully!")
        else:
            st.sidebar.error(f"Failed to load {selected_model_name}")
else:
    model = st.session_state.model

if uploaded_file is not None and 'model' in st.session_state:
    with col2:
        st.subheader("Detection Results")
        fig, class_counts, total_detections = process_image(uploaded_file, model, confidence)
        
        if fig:
            st.pyplot(fig)
            
            st.subheader("Detection Summary")
            
            if class_counts:
                summary_data = {
                    'Vehicle Type': list(class_counts.keys()),
                    'Count': list(class_counts.values())
                }
                
                summary_df = pd.DataFrame(summary_data)
                
                st.write(f"Total detections: {total_detections}")
                st.table(summary_df)
                
                if any(class_counts.values()):
                    st.subheader("Detection Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(class_counts.keys(), class_counts.values(), color='skyblue')
                    ax.set_xlabel('Vehicle Type')
                    ax.set_ylabel('Count')
                    ax.set_title('Detected Vehicles by Type')
                    st.pyplot(fig)
else:
    with col2:
        st.info("Upload an image to see detection results")
