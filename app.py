import tempfile
import av
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer,WebRtcMode
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="YOLO Object and Pose Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 2. CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    .hero-banner {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.2) 0%, rgba(14, 165, 233, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 700px;
        margin: 0 auto;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #0284c7) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    div[role="radiogroup"] {
        background: rgba(15, 23, 42, 0.6);
        padding: 10px 16px;
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        display: flex;
        justify-content: space-around;
    }

    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.5);
        border: 1.5px dashed #38bdf8;
        border-radius: 12px;
        padding: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.4);
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. LOAD YOLO MODELS (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------
@st.cache_resource
def load_detection_model():
  # YOLOv8 nano object detection model
  return YOLO("yolov8n.pt")


@st.cache_resource
def load_pose_model():
  # YOLOv8 nano pose estimation model
  return YOLO("yolov8n-pose.pt")


det_model = load_detection_model()
pose_model = load_pose_model()

# WebRTC STUN Server Config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
  st.markdown("Model Settings")
  st.markdown("Adjust inference thresholds in real time.")
  st.divider()

  conf_threshold = st.slider(
      "Confidence Threshold",
      min_value=0.10,
      max_value=0.95,
      value=0.40,
      step=0.05,
      help="Minimum confidence score required for bounding boxes & skeletons.",
  )

  st.markdown("Model Status")
  st.success("Models Loaded (yolov8n.pt , yolov8n-pose.pt)")
  st.divider()
  st.caption("By HAMZA ISLAM")

# ---------------------------------------------------------
# 5. HERO HEADER
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">VisionAI Multi-Modal Studio</div>
    <div class="hero-subtitle">
        Real-time YOLO Detection Portal for <b>Car & Person Detection</b> and <b>Human Pose Estimation</b> across Live Camera, Images, and Video Streams.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 6. TWO MAIN SECTIONS (TABS)
# ---------------------------------------------------------
tab_detection, tab_pose = st.tabs(
    ["Object Detection (Cars & Persons)", " Person Pose Detection"]
)


# --- TAB 1: CAR & PERSON OBJECT DETECTION ---
with tab_detection:
  st.markdown(
      """
    <div class="section-card">
        <h3 style="margin-top:0; color:#38bdf8;">Car & Person Object Detection</h3>
        <p style="color:#94a3b8; margin-bottom:0;">Target Classes: Person (ID: 0) & Car (ID: 2) | Engine: YOLOv8n</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  det_input_mode = st.radio(
      "Select Input Source for Object Detection:",
      options=["Real-Time Camera", "Picture Upload", "Video Upload"],
      horizontal=True,
      key="radio_det",
  )

  st.markdown("<br>", unsafe_allow_html=True)

  if det_input_mode == "Real-Time Camera":
    col1, col2 = st.columns([2, 1])
    with col1:
      st.markdown("#### Live Browser Webcam Object Detection")

      # Video callback for real-time WebRTC inference
      def det_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Run YOLOv8 detection filtering for person (0) and car (2)
        results = det_model(
            img, conf=conf_threshold, classes=[0, 2], verbose=False
        )
        annotated_img = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

      webrtc_streamer(
          key="webrtc_det",
          mode=WebRtcMode.SENDRECV,
          rtc_configuration=RTC_CONFIGURATION,
          video_frame_callback=det_frame_callback,
          media_stream_constraints={"video": True, "audio": False},
          async_processing=True,
      )
    with col2:
      st.markdown(
          """
            <div class="section-card">
                <h4 style="color:#e2e8f0; margin-top:0;">⚡ Live Inference</h4>
                <p style="font-size:0.9rem; color:#94a3b8;">
                    Click <b>START</b> to activate your camera. YOLOv8n will instantly annotate cars and people in real time.
                </p>
            </div>
            """,
          unsafe_allow_html=True,
      )

  elif det_input_mode == "Picture Upload":
    uploaded_img = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload_det",
    )

    if uploaded_img is not None:
      image = Image.open(uploaded_img)
      img_np = np.array(image)

      col1, col2 = st.columns(2)
      with col1:
        st.markdown("##### Original Image")
        st.image(image, width="stretch")

      with col2:
        st.markdown("##### YOLOv8 Detection Result")
        with st.spinner("Running detection..."):
          results = det_model(
              img_np, conf=conf_threshold, classes=[0, 2], verbose=False
          )
          res_plotted = results[0].plot()
          # Convert BGR (OpenCV) to RGB (Streamlit)
          res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
          st.image(res_rgb, width="stretch")

  elif det_input_mode == "Video Upload":
    uploaded_vid = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="vid_upload_det",
    )

    if uploaded_vid is not None:
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_vid.read())
      video_path = tfile.name

      col1, col2 = st.columns(2)
      with col1:
        st.markdown("##### Original Source Video")
        st.video(uploaded_vid)

      with col2:
        st.markdown("##### Processed Video Preview")
        if st.button("Run Detection on Video", key="btn_run_det_vid"):
          st.info(
              "Processing video frames... This may take a moment depending on"
              " video length."
          )
          cap = cv2.VideoCapture(video_path)
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          fps = cap.get(cv2.CAP_PROP_FPS)

          output_path = tempfile.NamedTemporaryFile(
              delete=False, suffix=".mp4"
          ).name
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

          progress_bar = st.progress(0)
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          frame_count = 0

          while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
              break
            results = det_model(
                frame, conf=conf_threshold, classes=[0, 2], verbose=False
            )
            res_frame = results[0].plot()
            out.write(res_frame)
            frame_count += 1
            if total_frames > 0:
              progress_bar.progress(
                  min(frame_count / total_frames, 1.0),
                  text=f"Processed frame {frame_count}/{total_frames}",
              )

          cap.release()
          out.release()
          progress_bar.empty()
          st.success("Video processing complete!")
          st.video(output_path)


# --- TAB 2: PERSON POSE DETECTION ---
with tab_pose:
  st.markdown(
      """
    <div class="section-card">
        <h3 style="margin-top:0; color:#38bdf8;">Human Skeletal Pose Estimation</h3>
        <p style="color:#94a3b8; margin-bottom:0;">Target: 17-Keypoint Full Body Skeleton Tracking | Engine: YOLOv8n-pose</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  pose_input_mode = st.radio(
      "Select Input Source for Pose Estimation:",
      options=["Real-Time Camera", "Picture Upload", "Video Upload"],
      horizontal=True,
      key="radio_pose",
  )

  st.markdown("<br>", unsafe_allow_html=True)

  if pose_input_mode == "Real-Time Camera":
    col1, col2 = st.columns([2, 1])
    with col1:
      st.markdown("#### Live Browser Webcam Pose Estimation")

      def pose_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Run YOLOv8 pose estimation
        results = pose_model(img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

      webrtc_streamer(
          key="webrtc_pose",
          mode=WebRtcMode.SENDRECV,
          rtc_configuration=RTC_CONFIGURATION,
          video_frame_callback=pose_frame_callback,
          media_stream_constraints={"video": True, "audio": False},
          async_processing=True,
      )
    with col2:
      st.markdown(
          """
            <div class="section-card">
                <h4 style="color:#e2e8f0; margin-top:0;">🏃 Skeletal Tracking</h4>
                <p style="font-size:0.9rem; color:#94a3b8;">
                    Click <b>START</b> to map joints and limbs live on your camera feed using `yolov8n-pose.pt`.
                </p>
            </div>
            """,
          unsafe_allow_html=True,
      )

  elif pose_input_mode == "Picture Upload":
    uploaded_pose_img = st.file_uploader(
        "Upload a person image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload_pose",
    )

    if uploaded_pose_img is not None:
      image = Image.open(uploaded_pose_img)
      img_np = np.array(image)

      col1, col2 = st.columns(2)
      with col1:
        st.markdown("##### Original Image")
        st.image(image, width="stretch")

      with col2:
        st.markdown("##### YOLOv8 Pose Estimation Result")
        with st.spinner("Extracting keypoints..."):
          results = pose_model(img_np, conf=conf_threshold, verbose=False)
          res_plotted = results[0].plot()
          res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
          st.image(res_rgb, width="stretch")

  elif pose_input_mode == "Video Upload":
    uploaded_pose_vid = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="vid_upload_pose",
    )

    if uploaded_pose_vid is not None:
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_pose_vid.read())
      video_path = tfile.name

      col1, col2 = st.columns(2)
      with col1:
        st.markdown("##### Original Source Video")
        st.video(uploaded_pose_vid)

      with col2:
        st.markdown("##### Pose Estimation Video Preview")
        if st.button("Run Pose Estimation on Video", key="btn_run_pose_vid"):
          st.info(
              "Processing video skeleton tracking... This may take a moment."
          )
          cap = cv2.VideoCapture(video_path)
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          fps = cap.get(cv2.CAP_PROP_FPS)

          output_path = tempfile.NamedTemporaryFile(
              delete=False, suffix=".mp4"
          ).name
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

          progress_bar = st.progress(0)
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          frame_count = 0

          while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
              break
            results = pose_model(frame, conf=conf_threshold, verbose=False)
            res_frame = results[0].plot()
            out.write(res_frame)
            frame_count += 1
            if total_frames > 0:
              progress_bar.progress(
                  min(frame_count / total_frames, 1.0),
                  text=f"Processed frame {frame_count}/{total_frames}",
              )

          cap.release()
          out.release()
          progress_bar.empty()
          st.success("Pose processing complete!")
          st.video(output_path)