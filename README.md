# FactLens: Advanced Video Text & Speech Metadata Extractor

FactLens is a modular, scalable, production-grade video metadata, speech-to-text, and visual text (OCR) extraction pipeline built in Python 3.12. It merges spoken words from audio with texts appearing on screen chronologically and stores the consolidated transcripts into MongoDB.

---

## Architecture Overview

```
Video
  │
  ├──► Audio → FFmpeg → WAV → Faster Whisper → Speech Text
  │
  ├──► Frames → OpenCV → JPG → EasyOCR → Screen Text
  │
  └──► Merge & Chronological Align
           │
           ▼
    Final Transcript
           │
           ▼
      MongoDB Database (RealityChecker)
```

---

## Directory Structure

```
factlens/
├── .env                  # Environment configurations (credentials, thresholds)
├── config.py             # Global settings and environment loading (Config class)
├── main.py               # Application orchestrator and CLI entrypoint
├── requirements.txt      # Python dependencies
├── README.md             # Installation, setup, and usage documentation
│
├── database/
│   ├── __init__.py       # Package exports
│   └── mongo_client.py   # OOP MongoDB client wrapper with indexing and CRUD operations
│
├── utils/
│   ├── __init__.py       # Package exports
│   ├── clean.py          # Space normalization, Unicode fixes, duplicate speech/OCR removal
│   ├── time_format.py    # Formatting utilities for HH:MM:SS or MM:SS conversion
│   └── video.py          # Video file validation and metadata properties extraction (OpenCV)
│
└── pipeline/
    ├── __init__.py       # Package exports
    ├── audio_extractor.py# Audio WAV extraction via FFmpeg subprocess call
    ├── transcriber.py    # Auto-language speech-to-text recognition via Faster Whisper
    ├── frame_extractor.py# Configurable frame sampling & extraction via OpenCV
    ├── ocr_engine.py     # Multi-lingual OCR text extraction via EasyOCR
    └── merger.py         # Chronological aligning of audio and screen texts
```

---

## Requirements

- **Python 3.12**
- **FFmpeg** (system-level executable)
- **MongoDB** (running instance)

---

## Installation & Setup

### 1. Install FFmpeg

FFmpeg is required for audio extraction.

- **Windows**:
  1. Download the build from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
  2. Extract the folder and add the `bin/` directory to your System Environment variables **`PATH`**.
  3. Verify by running `ffmpeg -version` in PowerShell or Command Prompt.

- **macOS**:
  ```bash
  brew install ffmpeg
  ```

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install ffmpeg -y
  ```

### 2. Set Up MongoDB

You need a running instance of MongoDB.
- You can run it locally or use a free cluster on MongoDB Atlas.
- Ensure the connection URI is specified in `.env` (default is `mongodb://localhost:27017/`).

### 3. Clone and Install Python Dependencies

1. Navigate to the project directory:
   ```bash
   cd C:\Users\rudra\.gemini\antigravity\scratch\factlens
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 4. Configure Environment Variables

Create or adjust the `.env` configuration file in the project root folder.

```ini
# FactLens Configuration
MONGO_URI=mongodb://localhost:27017/
FRAME_INTERVAL=1.0
OCR_THRESHOLD=0.4
TEMP_FOLDER=temp
WHISPER_MODEL=base
```

---

## Running the Project

To process a video file, execute the orchestrator script by passing the path of the target video file:

```bash
python main.py "path/to/your/video.mp4"
```

### Configurable CLI Flags

- `--interval`: Set custom frame extraction interval (e.g. `2.0` for 1 frame every 2 seconds).
  ```bash
  python main.py "video.mp4" --interval 2.0
  ```
- `--threshold`: Set custom EasyOCR confidence threshold (e.g. `0.5` to skip lower confidence).
  ```bash
  python main.py "video.mp4" --threshold 0.5
  ```

---

## Database Schema (RealityChecker.video_transcripts)

When processing completes, a document is stored in MongoDB inside the `RealityChecker` database and `video_transcripts` collection. Indexes are automatically created on `filename` and `created_at`.

```json
{
  "filename": "sample.mp4",
  "path": "C:\\Users\\rudra\\.gemini\\antigravity\\scratch\\factlens\\sample.mp4",
  "duration": 60.5,
  "fps": 30.0,
  "resolution": "1920x1080",
  "language": "en",
  "speech": [
    {
      "start": 0.0,
      "end": 3.4,
      "text": "Climate change is increasing.",
      "confidence": 0.985
    }
  ],
  "ocr": [
    {
      "timestamp": 2.0,
      "text": "GLOBAL TEMPERATURES",
      "confidence": 0.8741,
      "bbox": [[100, 200], [400, 200], [400, 250], [100, 250]]
    }
  ],
  "merged_transcript": [
    {
      "timestamp": 0.0,
      "timestamp_formatted": "00:00",
      "speech": "Climate change is increasing.",
      "visual": "GLOBAL TEMPERATURES"
    }
  ],
  "complete_text": "[00:00] Speech: \"Climate change is increasing.\" | Visual Text: [GLOBAL TEMPERATURES]",
  "created_at": "2026-06-26T14:00:00Z",
  "processing_time": "12.45s",
  "model": "faster-whisper-base"
}
```

---

## Log Output

Logs are outputted to the console and stored under the file **`factlens_processing.log`** in the project directory for tracking.
