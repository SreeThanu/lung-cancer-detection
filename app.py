import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from torchvision import models
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import io
import tempfile
import os
import sys

# Add project root to path to import from notebooks
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# MODEL DEFINITION (Same as your pre_trained_model.ipynb)
# ============================================================================
class LungCancerClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=False):
        super(LungCancerClassifier, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# ============================================================================
# PREPROCESSING FUNCTIONS (Same as your notebook)
# ============================================================================
def load_itk_image(filename):
    """Load CT scan using SimpleITK"""
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def normalize_hu(image, hu_min=-1000, hu_max=400):
    """Normalize Hounsfield Units"""
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image

def world_to_voxel(world_coords, origin, spacing):
    """Convert world coordinates to voxel coordinates"""
    stretched_voxel = np.absolute(world_coords - origin)
    voxel = stretched_voxel / spacing
    return voxel.astype(int)

def extract_nodule_patch(ct_scan, center, patch_size=64):
    """Extract 3D patch around nodule center"""
    z, y, x = center
    half_size = patch_size // 2
    
    z_start = max(0, z - half_size)
    z_end = min(ct_scan.shape[0], z + half_size)
    y_start = max(0, y - half_size)
    y_end = min(ct_scan.shape[1], y + half_size)
    x_start = max(0, x - half_size)
    x_end = min(ct_scan.shape[2], x + half_size)
    
    patch = ct_scan[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if patch.shape != (patch_size, patch_size, patch_size):
        padded = np.zeros((patch_size, patch_size, patch_size))
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded
    
    return patch

def extract_center_patch(ct_scan, patch_size=64):
    """Extract center patch from CT scan"""
    z, y, x = ct_scan.shape
    z_center, y_center, x_center = z // 2, y // 2, x // 2
    center = np.array([z_center, y_center, x_center])
    return extract_nodule_patch(ct_scan, center, patch_size)

def create_2d_projection(patch_3d, img_size=224):
    """Create 2D projection from 3D patch"""
    # Maximum Intensity Projection (MIP)
    mip = np.max(patch_3d, axis=0)
    
    # Resize to model input size
    zoom_factor = img_size / mip.shape[0]
    resized = zoom(mip, zoom_factor, order=1)
    
    # Convert to 3-channel (RGB)
    rgb_image = np.stack([resized] * 3, axis=0)
    
    return rgb_image

def preprocess_ct_scan(ct_file_path, coords=None, patch_size=64, img_size=224):
    """Full preprocessing pipeline from your notebook"""
    # Load CT scan
    ct_scan, origin, spacing = load_itk_image(ct_file_path)
    
    # Extract patch (either from coordinates or center)
    if coords is not None:
        # Use provided coordinates
        coord_x, coord_y, coord_z = coords
        world_coords = np.array([coord_z, coord_y, coord_x])
        voxel_coords = world_to_voxel(world_coords, origin, spacing)
        patch_3d = extract_nodule_patch(ct_scan, voxel_coords, patch_size)
    else:
        # Use center of scan
        patch_3d = extract_center_patch(ct_scan, patch_size)
    
    # Normalize
    patch_3d = normalize_hu(patch_3d)
    
    # Create 2D projection
    patch_2d = create_2d_projection(patch_3d, img_size)
    
    return patch_2d, ct_scan, patch_3d

# ============================================================================
# LOAD MODEL (From your trained checkpoint)
# ============================================================================
@st.cache_resource
def load_trained_model(model_path, model_name='resnet50', device='cpu'):
    """Load the trained model from checkpoint"""
    model = LungCancerClassifier(model_name=model_name, num_classes=2, pretrained=False)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, checkpoint.get('val_acc', 'N/A')
    else:
        return None, None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def visualize_ct_slices(ct_scan, num_slices=9):
    """Visualize multiple slices of CT scan"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    indices = np.linspace(0, ct_scan.shape[0]-1, num_slices, dtype=int)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            slice_img = ct_scan[indices[idx], :, :]
            ax.imshow(slice_img, cmap='gray')
            ax.set_title(f'Slice {indices[idx]}')
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_3d_patch(patch_3d):
    """Visualize 3D patch slices"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    mid_z = patch_3d.shape[0] // 2
    mid_y = patch_3d.shape[1] // 2
    
    # Axial slices
    for i in range(4):
        z_idx = max(0, min(patch_3d.shape[0]-1, mid_z - 15 + i * 10))
        axes[0, i].imshow(patch_3d[z_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Axial Slice {z_idx}')
        axes[0, i].axis('off')
    
    # Coronal slices
    for i in range(4):
        y_idx = max(0, min(patch_3d.shape[1]-1, mid_y - 15 + i * 10))
        axes[1, i].imshow(patch_3d[:, y_idx, :], cmap='gray')
        axes[1, i].set_title(f'Coronal Slice {y_idx}')
        axes[1, i].axis('off')
    
    plt.tight_tight()
    return fig

def visualize_2d_projection(patch_2d):
    """Visualize 2D projection"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(patch_2d[0], cmap='gray')
    ax.set_title('2D Maximum Intensity Projection (MIP)')
    ax.axis('off')
    plt.tight_layout()
    return fig

def create_prediction_gauge(probability):
    """Create a gauge chart for prediction probability"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    if probability < 0.33:
        color = '#2ecc71'
        risk_level = 'Low Risk'
    elif probability < 0.67:
        color = '#f39c12'
        risk_level = 'Medium Risk'
    else:
        color = '#e74c3c'
        risk_level = 'High Risk'
    
    ax.barh([0], [probability], color=color, height=0.5, alpha=0.8)
    ax.barh([0], [1-probability], left=probability, color='lightgray', height=0.5, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Cancer Detection Probability: {probability:.2%} - {risk_level}', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    # Add risk zone markers
    ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.67, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add percentage text
    ax.text(probability/2, 0, f'{probability:.1%}', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Lung Cancer Detection System",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .positive {
            background-color: #ffebee;
            border-left: 5px solid #e74c3c;
        }
        .negative {
            background-color: #e8f5e9;
            border-left: 5px solid #2ecc71;
        }
        .stProgress > div > div > div > div {
            background-color: #2ecc71;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("### AI-Powered CT Scan Analysis using Deep Learning")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        model_path = st.text_input(
            "Model Checkpoint Path", 
            value="best_lung_cancer_model.pth",
            help="Path to your trained model from pre_trained_model.ipynb"
        )
        
        model_name = st.selectbox(
            "Model Architecture",
            ["resnet50", "resnet101", "densenet121", "efficientnet_b0"],
            help="Must match the architecture used in training"
        )
        
        # Prediction settings
        st.subheader("🎯 Prediction Settings")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for positive classification"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_coordinates = st.checkbox(
                "Use Nodule Coordinates",
                value=False,
                help="If checked, provide nodule coordinates instead of using center patch"
            )
            
            if use_coordinates:
                coord_x = st.number_input("Coordinate X", value=0.0)
                coord_y = st.number_input("Coordinate Y", value=0.0)
                coord_z = st.number_input("Coordinate Z", value=0.0)
                coords = (coord_x, coord_y, coord_z)
            else:
                coords = None
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This system uses the pre-trained model from your 
        **pre_trained_model.ipynb** notebook to detect 
        potential lung cancer from CT scans.
        
        **Supported formats:** .mhd + .raw
        
        **Model:** Transfer learning with fine-tuned 
        deep neural networks on LUNA16 dataset.
        
        ⚠️ **Disclaimer:** For research purposes only.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload CT Scan")
        
        uploaded_file = st.file_uploader(
            "Choose a CT scan file (.mhd)",
            type=['mhd'],
            help="Upload the .mhd header file of your CT scan"
        )
        
        uploaded_raw = st.file_uploader(
            "Choose the corresponding .raw file",
            type=['raw'],
            help="Upload the .raw data file that corresponds to your .mhd file"
        )
        
        st.markdown("---")
        analyze_button = st.button(
            "🔍 Analyze CT Scan", 
            type="primary", 
            use_container_width=True,
            disabled=(uploaded_file is None or uploaded_raw is None)
        )
    
    with col2:
        st.header("📋 Quick Guide")
        st.markdown("""
        **Step-by-step:**
        1. 📁 Upload both `.mhd` and `.raw` files from your CT scan
        2. ⚙️ Configure model path in sidebar (default: `best_lung_cancer_model.pth`)
        3. 🎯 Adjust classification threshold if needed (default: 0.5)
        4. 🔍 Click "Analyze CT Scan" button
        5. 📊 Review results and visualizations
        
        **Note:** Make sure your model checkpoint from 
        `pre_trained_model.ipynb` exists at the specified path.
        """)
        
        # Display model info if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"**Compute Device:** {device}")
    
    st.markdown("---")
    
    # Analysis Section
    if analyze_button:
        if uploaded_file is None or uploaded_raw is None:
            st.error("❌ Please upload both .mhd and .raw files!")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load model
            status_text.text("🔄 Loading trained model...")
            progress_bar.progress(10)
            
            model, val_acc = load_trained_model(model_path, model_name, device)
            
            if model is None:
                st.error(f"❌ Model file not found at: `{model_path}`")
                st.info("💡 Make sure you've trained the model using `pre_trained_model.ipynb` first!")
                return
            
            st.success(f"✅ Model loaded successfully! (Validation Accuracy: {val_acc})")
            progress_bar.progress(25)
            
            # Step 2: Process CT scan
            status_text.text("🔄 Processing CT scan files...")
            progress_bar.progress(40)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                mhd_path = os.path.join(tmpdir, uploaded_file.name)
                raw_path = os.path.join(tmpdir, uploaded_raw.name)
                
                with open(mhd_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                with open(raw_path, 'wb') as f:
                    f.write(uploaded_raw.getbuffer())
                
                progress_bar.progress(55)
                status_text.text("🔄 Extracting features and creating projections...")
                
                # Preprocess using the same pipeline as your notebook
                patch_2d, ct_scan, patch_3d = preprocess_ct_scan(
                    mhd_path, 
                    coords=coords if use_coordinates else None
                )
            
            st.success("✅ CT scan processed successfully!")
            progress_bar.progress(70)
            
            # Step 3: Run inference
            status_text.text("🔄 Running AI model inference...")
            progress_bar.progress(85)
            
            input_tensor = torch.FloatTensor(patch_2d).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                cancer_prob = probabilities[0, 1].item()
                healthy_prob = probabilities[0, 0].item()
                predicted_class = 1 if cancer_prob >= threshold else 0
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Prediction",
                    "CANCER" if predicted_class == 1 else "HEALTHY",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "📈 Cancer Probability",
                    f"{cancer_prob:.2%}",
                    delta=f"{cancer_prob - threshold:.2%}"
                )
            
            with col3:
                st.metric(
                    "💚 Healthy Probability",
                    f"{healthy_prob:.2%}",
                    delta=None
                )
            
            with col4:
                risk = "🔴 High" if cancer_prob > 0.67 else "🟡 Medium" if cancer_prob > 0.33 else "🟢 Low"
                st.metric(
                    "⚠️ Risk Level",
                    risk.split()[1],
                    delta=None
                )
            
            # Detailed result box
            result_class = "positive" if predicted_class == 1 else "negative"
            result_icon = "⚠️" if predicted_class == 1 else "✅"
            result_text = "Cancer Detected" if predicted_class == 1 else "No Cancer Detected"
            
            st.markdown(f"""
                <div class="result-box {result_class}">
                    <h2>{result_icon} {result_text}</h2>
                    <h3>Detailed Probabilities:</h3>
                    <ul>
                        <li><strong>Cancer Probability:</strong> {cancer_prob:.4f} ({cancer_prob:.2%})</li>
                        <li><strong>Healthy Probability:</strong> {healthy_prob:.4f} ({healthy_prob:.2%})</li>
                        <li><strong>Threshold Used:</strong> {threshold}</li>
                        <li><strong>Model:</strong> {model_name}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("### 📊 Probability Visualization")
            gauge_fig = create_prediction_gauge(cancer_prob)
            st.pyplot(gauge_fig)
            plt.close()
            
            # ================================================================
            # VISUALIZATIONS
            # ================================================================
            st.markdown("---")
            st.header("🔬 Medical Imaging Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📸 CT Scan Slices", 
                "🧊 3D Patch Analysis", 
                "🎯 2D Projection (Model Input)", 
                "ℹ️ Scan Metadata"
            ])
            
            with tab1:
                st.markdown("#### Complete CT Scan - Multiple Axial Slices")
                st.write("These are evenly-spaced slices through the entire CT volume:")
                ct_fig = visualize_ct_slices(ct_scan)
                st.pyplot(ct_fig)
                plt.close()
            
            with tab2:
                st.markdown("#### Extracted 3D Patch - Orthogonal Views")
                st.write("Different views of the 64x64x64 patch extracted for analysis:")
                patch_fig = visualize_3d_patch(patch_3d)
                st.pyplot(patch_fig)
                plt.close()
            
            with tab3:
                st.markdown("#### 2D Maximum Intensity Projection")
                st.write("This is the actual input fed to the neural network:")
                projection_fig = visualize_2d_projection(patch_2d)
                st.pyplot(projection_fig)
                plt.close()
            
            with tab4:
                st.markdown("#### Scan Information & Metadata")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📏 CT Scan Dimensions:**")
                    st.write(f"- **Full Shape:** {ct_scan.shape}")
                    st.write(f"- **Total Slices (Z):** {ct_scan.shape[0]}")
                    st.write(f"- **Height (Y):** {ct_scan.shape[1]}")
                    st.write(f"- **Width (X):** {ct_scan.shape[2]}")
                    st.write(f"- **Total Voxels:** {np.prod(ct_scan.shape):,}")
                
                with col2:
                    st.markdown("**⚙️ Processing Parameters:**")
                    st.write(f"- **Patch Size:** 64×64×64 voxels")
                    st.write(f"- **Projection Size:** 224×224 pixels")
                    st.write(f"- **Model Architecture:** {model_name}")
                    st.write(f"- **Compute Device:** {device}")
                    st.write(f"- **Coordinates Used:** {'Yes' if use_coordinates else 'No (Center)'}")
                
                st.markdown("---")
                st.markdown("**📊 Hounsfield Unit Statistics:**")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min HU", f"{ct_scan.min():.1f}")
                col2.metric("Max HU", f"{ct_scan.max():.1f}")
                col3.metric("Mean HU", f"{ct_scan.mean():.1f}")
                col4.metric("Std HU", f"{ct_scan.std():.1f}")
            
            # ================================================================
            # INTERPRETATION GUIDE
            # ================================================================
            st.markdown("---")
            st.header("📖 Understanding Your Results")
            
            with st.expander("💡 How to interpret the prediction", expanded=False):
                st.markdown("""
                **Cancer Probability Ranges:**
                - **0% - 33%:** Low risk - Model suggests healthy tissue with high confidence
                - **34% - 66%:** Medium risk - Uncertain prediction, recommend further investigation
                - **67% - 100%:** High risk - Model suggests potential cancer with high confidence
                
                **Important Notes:**
                - This model was trained on LUNA16 dataset for nodule detection
                - Predictions should be validated by medical professionals
                - False positives and false negatives are possible
                - Use this as a screening tool, not a diagnostic tool
                """)
            
            with st.expander("🔬 About the Model Architecture", expanded=False):
                st.markdown(f"""
                **Model Details:**
                - **Base Architecture:** {model_name} (Pre-trained on ImageNet)
                - **Training Dataset:** LUNA16 Subset 0
                - **Transfer Learning:** Fine-tuned on lung CT scans
                - **Input:** 224×224 RGB projection from 3D CT patches
                - **Output:** Binary classification (Cancer / No Cancer)
                - **Validation Accuracy:** {val_acc}
                
                The model uses Maximum Intensity Projection (MIP) to convert 3D CT patches 
                into 2D images suitable for pre-trained 2D CNNs.
                """)
            
            # ================================================================
            # DISCLAIMER
            # ================================================================
            st.markdown("---")
            st.warning("""
            ⚕️ **IMPORTANT MEDICAL DISCLAIMER**
            
            This AI system is designed for **research and educational purposes only**. 
            
            - **Not FDA approved** for clinical use
            - **Not a substitute** for professional medical diagnosis
            - Results should be **interpreted by qualified radiologists**
            - Always **consult healthcare professionals** for medical decisions
            - False positives and false negatives can occur
            
            If you have health concerns, please seek immediate medical attention from licensed healthcare providers.
            """)
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            with st.expander("🔍 View Full Error Details"):
                st.exception(e)
            
            st.info("""
            **Common issues:**
            - Model checkpoint not found → Train model using `pre_trained_model.ipynb` first
            - File format errors → Ensure you're uploading valid .mhd + .raw pairs
            - Memory errors → Try reducing batch size or using smaller CT scans
            """)

if __name__ == "__main__":
    main()