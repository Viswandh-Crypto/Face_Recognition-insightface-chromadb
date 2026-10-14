# 👤 Face Recognition with InsightFace and ChromaDB

A face-recognition prototype built in **Google Colab** using OpenCV Haar Cascade, InsightFace, and ChromaDB. This project detects faces in a group photo, generates face embeddings, stores them in a vector database, and searches for the closest matching face when a query image is uploaded.

> ⚠️ **Project Status:** Prototype / Learning Project. The similarity threshold is experimental, and results should not be treated as proof of a person's identity.

## 📌 Table of Contents

- [🌟 Overview](#-overview)
- [🔄 Project Workflow](#-project-workflow)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [🧠 How It Works](#-how-it-works)
- [📏 Cosine Distance and Matching](#-cosine-distance-and-matching)
- [💾 Data Storage and Persistence](#-data-storage-and-persistence)
- [⚠️ Current Limitations](#️-current-limitations)
- [🔮 Future Improvements](#-future-improvements)
- [📁 Repository Structure](#-repository-structure)
- [🔐 Before Publishing](#-before-publishing-to-github)
- [🛡️ Responsible Use](#️-responsible-use)
- [🙏 Acknowledgements](#-acknowledgements)

## 🌟 Overview

This project combines traditional computer vision, deep-learning-based face analysis, and vector similarity search.

- 🔍 **OpenCV Haar Cascade:** Detects frontal-face regions in the original group photo.
- ✂️ **OpenCV:** Draws bounding boxes and crops detected faces.
- 🧠 **InsightFace (`buffalo_l`):** Detects faces in query images and generates normalized face embeddings.
- 🗄️ **ChromaDB:** Stores embeddings and retrieves nearest candidates using cosine distance.
- 🔢 **NumPy:** Handles numerical operations and embedding data.
- 📊 **Matplotlib:** Visualizes detected faces and query results.
- ☁️ **Google Colab:** Provides the notebook execution environment.

In the current implementation, Haar Cascade is used for initial group-photo detection and cropping. InsightFace is used to generate embeddings and detect faces in uploaded query images.

## 🔄 Project Workflow

![Face Recognition Project Workflow](assets/face-recognition-workflow.png)

The project consists of two main stages:

### 1️⃣ Face Database Creation

1. 📤 Upload a group photo in Google Colab.
2. 🖼️ Convert the image to grayscale and apply histogram equalization.
3. 🔍 Detect frontal faces using OpenCV Haar Cascade.
4. 🟩 Draw bounding boxes around detected faces.
5. ✂️ Crop and save detected faces in the `stored-faces/` directory.
6. 🧠 Initialize InsightFace using the `buffalo_l` model.
7. 🔢 Generate a normalized, 512-dimensional embedding for each usable face crop.
8. 🗄️ Store embeddings and source filenames in a ChromaDB collection named `face_embeddings`.

### 2️⃣ Face Search and Matching

1. 📤 Upload a query photo.
2. 🔍 Use InsightFace to detect faces in the query image.
3. 👤 Select the largest detected face when multiple faces are present.
4. 🧠 Generate a normalized face embedding.
5. 🔎 Search ChromaDB for the three nearest candidates.
6. 📏 Compare the closest candidate's cosine distance with the experimental threshold of `0.45`.
7. ✅ Display a candidate match if it passes the threshold; otherwise, report **No Confident Match**.

> 📝 *Note: The workflow diagram is illustrative. Example faces, embedding values, and candidate distances shown in the diagram are not guaranteed to represent actual notebook outputs.*

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| ☁️ Google Colab | Notebook execution |
| 👁️ OpenCV | Image preprocessing, face cropping, and visualization |
| 🔍 Haar Cascade | Initial frontal-face detection |
| 🧠 InsightFace (`buffalo_l`) | Face analysis and embedding generation |
| ⚡ ONNX Runtime | Model inference |
| 🗄️ ChromaDB | Vector storage and similarity search |
| 🔢 NumPy | Numerical operations |
| 📊 Matplotlib | Visualization |

## 🚀 Getting Started

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/face-recognition-insightface-chromadb.git
cd face-recognition-insightface-chromadb
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. ☁️ Open the Notebook

Open `Face_Recognition.ipynb` in Google Colab.

Alternatively, upload the notebook manually to Google Colab.

### 3. 📦 Install Dependencies

The notebook installs its dependencies in Colab. If you need to install them manually, run:

```python
%pip install opencv-python-headless==4.10.0.84
%pip install insightface onnxruntime
%pip install chromadb numpy matplotlib
```

> 💡 Package compatibility may vary depending on the Python version and dependency versions. If installation fails, check the relevant package requirements.

### 4. ▶️ Run the Notebook

Execute the notebook cells in order:

1. Install dependencies and import libraries.
2. Upload the original group photo.
3. Run Haar Cascade detection and face cropping.
4. Initialize InsightFace and generate embeddings.
5. Initialize ChromaDB and store the embeddings.
6. Upload a query photo in the face-search section.
7. Review the closest candidates and the threshold decision.

The current notebook uses CPU execution for InsightFace. A GPU is not required.

## 🧠 How It Works

### 🔍 Face Detection

OpenCV Haar Cascade detects frontal-face regions in the original group photo. Detection parameters such as `scaleFactor`, `minNeighbors`, and `minSize` control its behavior.

### ✂️ Face Cropping

OpenCV extracts detected face regions and saves them as individual images in the `stored-faces/` directory.

### 🔢 Face Embedding Generation

InsightFace generates normalized, 512-dimensional numerical representations of detected faces. These embeddings allow the system to compare facial features mathematically.

### 🗄️ Vector Storage and Search

ChromaDB stores the face embeddings and associated source filenames. When a query embedding is submitted, ChromaDB retrieves the nearest candidates using cosine distance.

## 📏 Cosine Distance and Matching

ChromaDB uses cosine distance to compare the query embedding with stored embeddings.

- 📉 **Lower distance:** More similar embeddings.
- 📈 **Higher distance:** Less similar embeddings.
- ✅ **Distance ≤ 0.45:** Candidate passes the current experimental threshold.
- ❌ **Distance > 0.45:** Candidate is rejected as a confident match.

The threshold of `0.45` is a starting point for experimentation, not a universal InsightFace threshold. It should be calibrated using representative known matches and non-matches.

⚠️ A candidate passing the threshold does not guarantee that the two images belong to the same person.

## 💾 Data Storage and Persistence

The notebook generates the following local data:

```text
stored-faces/
    face_0.jpg
    face_1.jpg
    face_2.jpg
    ...

/content/chroma_db/
    ChromaDB database files
```

The ChromaDB collection is named `face_embeddings` and uses cosine distance.

Google Colab's local filesystem is temporary. Files stored under `/content/` may be lost when the runtime is reset or disconnected.

The notebook includes a backup step that copies the database to Google Drive:

```text
MyDrive/FaceRecognitionProject/chroma_db/
```

To retain the database across sessions:

1. 🔗 Mount Google Drive in Colab.
2. 💾 Run the database backup cell after creating or updating the database.
3. 🔄 In a new runtime, restore or configure the database path and ensure the face crops are available before querying.

> 📌 The local database path and Google Drive backup path are separate. A backup is not automatic unless the relevant notebook cell is executed.

## ⚠️ Current Limitations

- Haar Cascade primarily detects frontal faces and may miss faces under challenging poses, lighting, or occlusion.
- Group-photo detection uses Haar Cascade, while query-image detection uses InsightFace. Differences between these detectors may affect crop consistency.
- The query pipeline selects the largest detected face. If the uploaded image contains multiple people, the selected face may not be the intended person.
- The database stores cropped-face embeddings and source filenames. It does not yet fully map every stored crop to its original group-photo bounding box for highlighting the matched person in the group image.
- A nearest-neighbor search returns candidates even when none is a genuine match. Thresholding helps reject poor candidates but cannot guarantee identity.
- The `0.45` cosine-distance threshold is experimental and requires validation.
- This implementation is a notebook prototype, not a production-ready identity-verification system.

## 🔮 Future Improvements

- 🔄 Use InsightFace for both group-photo detection and query-image detection.
- 📍 Preserve original-image bounding boxes and metadata for every stored face.
- 🖼️ Display the uploaded query photo, matched crop, and original group photo with the corresponding face highlighted.
- 📏 Calibrate the distance threshold using a representative validation dataset.
- 👥 Improve handling of images containing multiple people.
- 🛠️ Add structured logging and clearer error handling.
- ☁️ Improve database persistence and reproducibility across Colab sessions.

## 📁 Repository Structure

```text
face-recognition-insightface-chromadb/
├── Face_Recognition.ipynb
├── README.md
├── .gitignore
└── assets/
    └── face-recognition-workflow.png
```

The `stored-faces/` directory and ChromaDB database can be generated by running the notebook. They generally should not be committed to a public repository.

## 🔐 Before Publishing to GitHub

- 🧹 Clear notebook outputs that contain private photos or information you do not want to publish.
- 🚫 Do not commit private query photos, face crops, or database files.
- 🔑 Check the notebook for API keys, credentials, and other secrets.
- 🖼️ Ensure you have permission to share any images included in the repository.
- 🛡️ Add generated data and notebook checkpoints to `.gitignore`.

Example `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Generated face crops
stored-faces/

# Local vector database
chroma_db/
*.sqlite3
*.db

# Local query images
query-*.jpg
query-*.jpeg
query-*.png

# Notebook checkpoints
.ipynb_checkpoints/
```

## 🛡️ Responsible Use

Use this project only with images you have permission to process. Face images and face embeddings may constitute sensitive biometric data.

- Obtain appropriate consent where required.
- 🔒 Protect stored images and embeddings.
- ⚖️ Follow applicable privacy and data-protection requirements.
- 🚫 Do not use this prototype as the sole basis for consequential decisions.

This project is intended for educational purposes and experimentation with computer vision, face embeddings, and vector similarity search.

## 🙏 Acknowledgements

- 🧠 [InsightFace](https://github.com/deepinsight/insightface)
- 👁️ [OpenCV](https://opencv.org/)
- 🗄️ [ChromaDB](https://www.trychroma.com/)
- ☁️ [Google Colab](https://colab.research.google.com/)

---

⭐ If you find this project useful, feel free to explore it, learn from it, and build upon it!
