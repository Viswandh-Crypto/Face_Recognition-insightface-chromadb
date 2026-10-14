# Face Recognition with InsightFace and ChromaDB

A Google Colab project that detects faces in a group photo, creates face crops, generates face embeddings with InsightFace, stores the embeddings in ChromaDB, and searches for the closest stored face when a query photo is uploaded.

> **Project status:** Prototype / learning project. The similarity threshold is experimental and results should not be treated as proof of a person's identity.

## Overview

This project combines traditional computer vision with deep-learning-based face analysis:

- **OpenCV Haar Cascade** detects frontal-face regions in the original group photo.
- **OpenCV** crops and saves each detected face.
- **InsightFace (`buffalo_l`)** detects faces in the crops and query images and generates normalized face embeddings.
- **ChromaDB** stores embeddings and retrieves nearest candidates using cosine distance.
- **Matplotlib** displays detected faces and query results.

In the current notebook, Haar Cascade is used for the initial group-photo detection and cropping. InsightFace is used for embedding generation and for detecting the face in an uploaded query image.

## Workflow

1. Upload a group photo in Google Colab.
2. Convert the image to grayscale and equalize its histogram.
3. Detect frontal faces with Haar Cascade.
4. Draw bounding boxes and save detected face crops to `stored-faces/`.
5. Initialize InsightFace and generate a normalized embedding for each usable crop.
6. Store the embeddings and source filenames in a ChromaDB collection named `face_embeddings`.
7. Upload a query photo.
8. Use InsightFace to detect a face in the query image and generate its embedding.
9. Search ChromaDB for the three nearest candidates.
10. Compare the closest result's cosine distance with an experimental threshold (`0.45`) and report either a candidate match or no confident match.

**Cosine distance interpretation:** lower values indicate more similar embeddings; higher values indicate less similarity. The threshold of `0.45` is only a starting point and must be calibrated against representative positive and negative test images.

## Tech Stack

- Python
- Google Colab
- OpenCV (`opencv-python-headless`)
- InsightFace (`buffalo_l`)
- ONNX Runtime
- ChromaDB
- NumPy
- Matplotlib

## Run the Notebook

1. Open the notebook in Google Colab.
2. Run the setup and import cells.
3. Upload a group photo when prompted.
4. Run the Haar Cascade detection and face-cropping cells.
5. Initialize InsightFace and generate embeddings.
6. Initialize ChromaDB and run the embedding-storage cell.
7. Upload a query photo in the query section to search for similar stored faces.

The notebook currently uses CPU execution for InsightFace (`CPUExecutionProvider`), so a GPU is not required.

## Installation Notes

The notebook installs dependencies in Colab. The key packages used are:

```bash
pip install opencv-python-headless==4.10.0.84
pip install insightface onnxruntime
pip install chromadb
```

Other imports used include `numpy` and `matplotlib`, which are typically available in Colab. Package compatibility can change over time; if a fresh runtime produces dependency errors, check the installed Python version and package compatibility.

## Data and Persistence

The notebook creates a local folder named `stored-faces/` for face crops and a ChromaDB database at:

```text
/content/chroma_db
```

These paths are on the Colab runtime filesystem. **They are not guaranteed to survive a runtime reset.** The notebook includes a cell that copies the database to Google Drive under:

```text
MyDrive/FaceRecognitionProject/chroma_db
```

Run that backup cell after the database has been created or updated. For a fresh session, make sure the face crops and database are available before running the query cells.

## Current Limitations

- Haar Cascade is primarily suited to frontal faces and can miss faces or produce false detections depending on pose, lighting, image quality, and occlusion.
- The initial group-photo detector (Haar Cascade) differs from the query detector (InsightFace). This can lead to inconsistent crops and is a possible area for improvement.
- The notebook selects the largest detected face in a query photo. Photos containing multiple people may therefore query the wrong person.
- The database currently stores cropped-face embeddings and source filenames; it does not yet fully map every crop back to its original group-photo bounding box for highlighting the person in the original image.
- A nearest neighbor is always returned when the collection is non-empty. The distance threshold helps reject poor candidates, but it is not a guarantee of identity.
- The `0.45` threshold is experimental, not a universal InsightFace threshold. Evaluate it with multiple known matches and non-matches before relying on it.
- This is a notebook prototype, not a production-ready identity-verification or access-control system.

## Suggested Repository Structure

```text
face-recognition-insightface-chromadb/
├── Face_Recognition.ipynb
├── README.md
├── .gitignore
└── stored-faces/              # optional local data; do not commit private face images
```

The `stored-faces/` folder and ChromaDB files can be generated locally by running the notebook. Avoid committing personal or copyrighted images unless you have permission to share them.

## Before Pushing to GitHub

- Clear notebook outputs if they contain personal photos, large embedded images, local paths, or other information you do not want to publish.
- Do not upload private query photos, face crops, or the local ChromaDB database unless you intentionally want to make that data public and have permission.
- Check that the notebook contains no API keys, credentials, or other secrets.
- Add generated files and local data to `.gitignore` as appropriate.

Example `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Generated face crops and local vector database
stored-faces/
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

## Responsible Use

Use this project only with images you have the right to process. Face embeddings and face images can be sensitive biometric data. Store them securely, obtain appropriate consent where required, and do not use this prototype as the sole basis for consequential decisions.

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface)
- [OpenCV](https://opencv.org/)
- [Chroma](https://www.trychroma.com/)
- [Google Colab](https://colab.research.google.com/)
