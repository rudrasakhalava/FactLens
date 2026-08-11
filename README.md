# FactLens: Multimodal AI Video Fact-Checking System

FactLens is a **multimodal AI-powered video fact-checking system** that analyzes video content, extracts speech and on-screen text, reconstructs multimodal context, identifies factual claims, researches those claims using multiple web sources, and generates evidence-based results with voice output.

The project was initially developed as a traditional sequential video metadata and transcription pipeline. It has now been redesigned using **LangGraph** to provide a modular, state-driven workflow with parallel processing, conditional data flow, and specialized AI processing nodes.

---

## 🚀 Key Features

- 🎥 Video metadata extraction
- 🎙️ Automatic speech transcription
- 🌐 Multi-model audio language detection
- 📝 Multilingual OCR extraction
- 🔤 OCR language detection
- 🧠 AI-powered video description
- 🔄 Context reconstruction from fragmented speech
- 🌍 Translation of extracted content to English
- 🔗 Multimodal context fusion
- 📌 Automatic factual claim extraction
- 🔎 Multi-source web research
- 📰 Google News search
- 🌐 Wikipedia search
- 🔍 DuckDuckGo web search
- 🤖 LLM-powered claim analysis
- 🌎 Translation of final results
- 🔊 Text-to-speech output
- ⚡ Parallel execution using LangGraph
- 🧩 Modular state-based architecture

---

# 🏗️ Architecture

FactLens uses **LangGraph** to orchestrate the complete video analysis and fact-checking workflow.

The workflow can be divided into six major stages:

```text
Video
  │
  ▼
Metadata Extraction
  │
  ├──────────────────────┬──────────────────────┐
  ▼                      ▼                      ▼
Video Description    OCR Language         Audio Language
                     Detection             Detection
                                             │
                                  ┌──────────┴──────────┐
                                  ▼                     ▼
                              MMS-LID               Whisper
                                  │                     │
                                  └──────────┬──────────┘
                                             ▼
                                     Final Audio Selection
                                             │
                                             ▼
                                      Audio Transcription
                                             │
                                             ▼
                                       Rebuild Context
                                             │
                                             ▼
                                      Translate to English
                                             │
                                             ├───────────────┐
                                             │               │
                                             ▼               ▼
                                      OCR / Visual     Video Description
                                             │               │
                                             └───────┬───────┘
                                                     ▼
                                          Merge Multimodal Context
                                                     │
                                                     ▼
                                              Extract Claims
                                                     │
                           ┌─────────────────┬────────┴─────────┐
                           ▼                 ▼                  ▼
                       DDGS Search     Google News Search   Wikipedia Search
                           │                 │                  │
                           └─────────────────┴────────┬─────────┘
                                                    ▼
                                         Merge Research Results
                                                    │
                                                    ▼
                                             Merge Results
                                                    │
                                                    ▼
                                             Translate Text
                                                    │
                                                    ▼
                                             Text-to-Speech
                                                    │
                                                    ▼
                                                   END
