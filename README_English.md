# Face Swap Web Application

## Overview

This project is a Flask-based web application designed to detect and distinguish faces in videos, providing a foundation for face-swapping operations. It leverages MTCNN for face detection and InceptionResnetV1 for face recognition, with GPU acceleration via CUDA for enhanced performance. Users can upload videos through a user-friendly interface to extract and save distinct faces.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [GPU and CUDA Support](#gpu-and-cuda-support)
- [Function Details](#function-details)
- [Project Structure](#project-structure)
- [Frequently Asked Questions](#frequently-asked-questions)
- [License](#license)
- [Contact](#contact)

---

## Features

- Detection and differentiation of distinct faces in videos
- High-accuracy face detection using MTCNN (GPU-supported)
- Face recognition and comparison with InceptionResnetV1
- User-friendly Flask-based web interface
- Progress tracking via `progress_status`
- Automatic saving of detected faces to an output folder
- GPU acceleration with CUDA support, with fallback to CPU
- Flexible configuration: customizable maximum frame count and tolerance parameters

---

## Technologies Used

- **Python**: 3.8 or higher
- **Flask**: Web application framework
- **PyTorch**: Deep learning models (CUDA-supported recommended)
- **facenet-pytorch**: MTCNN and InceptionResnetV1 models
- **OpenCV**: Image and video processing
- **Pillow (PIL)**: Image manipulation
- **NumPy**: Mathematical operations

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/face-swap-web.git
cd face-swap-web
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies

For CUDA-supported PyTorch (e.g., CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For systems without CUDA support:

```bash
pip install torch torchvision torchaudio
```

Other dependencies:

```bash
pip install facenet-pytorch opencv-python flask pillow numpy
```

**Note**: Ensure the PyTorch version matches your system's CUDA version. Check your CUDA version using `nvidia-smi`.

---

## Usage

### 1. Start the Flask Server

```bash
# Linux/macOS
export FLASK_APP=app.py
export FLASK_ENV=development
flask run

# Windows PowerShell
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
flask run
```

### 2. Upload a Video

- Navigate to the web interface (default: `http://127.0.0.1:5000`).
- Upload a video file.
- The application detects faces in the video and saves distinct faces to the `output_faces` folder.

### 3. Outputs

- Detected faces are saved as PNG files in the `output_faces` folder.
- Frame numbers and face position information are returned for each face.

---

## GPU and CUDA Support

The application checks for GPU availability at startup. If CUDA is available, models run on the GPU; otherwise, they fall back to CPU. The GPU status is displayed in the console as follows:

```yaml
CUDA Available: Yes
Total GPUs: 1
GPU 0: NVIDIA GeForce RTX 3060
Device: cuda:0
```

**Note**: CUDA support requires an NVIDIA GPU with compatible drivers installed.

---

## Function Details

### `extract_distinct_faces(video_path, output_folder, max_frames=50, tolerance=0.6, progress_status=None)`

#### Parameters

- `video_path` (str): Path to the input video file.
- `output_folder` (str): Directory where detected faces are saved.
- `max_frames` (int): Maximum number of frames to process (default: 50).
- `tolerance` (float): Face recognition threshold (default: 0.6).
- `progress_status` (dict, optional): Dictionary for tracking process progress.

#### Workflow

1. The video is processed frame by frame (up to `max_frames`).
2. MTCNN performs face detection.
3. Detected faces are embedded using InceptionResnetV1.
4. New faces are compared against previous ones using the tolerance threshold.
5. Unique faces are saved to the `output_folder`.
6. Progress is updated in the `progress_status` dictionary.

#### Return Value

- `face_map`: A dictionary containing face IDs, frame numbers, and face position information.

---

## Project Structure

```plaintext
face-swap-web/
│
├── app.py                # Main Flask application file
├── face_extractor.py     # Face detection and recognition functions
├── templates/            # HTML templates
│   └── index.html        # Main web interface
├── static/               # CSS, JS, and uploaded files
│   ├── css/
│   ├── js/
│   └── uploads/          # Uploaded videos
├── output_faces/         # Folder for saved face images
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## Frequently Asked Questions (FAQ)

**Q: How do I enable CUDA support?**  
A: Ensure an NVIDIA GPU and compatible CUDA drivers are installed. Use the CUDA-supported PyTorch version.

**Q: How accurate is face detection?**  
A: MTCNN provides high accuracy, but performance may degrade with low-resolution videos or poor lighting.

**Q: How can I change the maximum frame count?**  
A: Adjust the `max_frames` parameter in the `extract_distinct_faces` function.

**Q: Does this include face-swapping functionality?**  
A: This project focuses on face detection and recognition. Face swapping requires additional modules.

**Q: Which video formats are supported?**  
A: All formats supported by OpenCV (e.g., MP4, AVI) are compatible.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Thank you!

---