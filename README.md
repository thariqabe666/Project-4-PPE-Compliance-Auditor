# Automated PPE Compliance Auditor (YOLOv12 + Streamlit)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

<p align="center">
  <img src="docs/demo.gif" alt="App Demo" width="800"/>
</p>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ppe-hse-compliance-auditor.streamlit.app/)

## 🚀 About This Project

This project is an AI-powered application designed to automatically audit **Personal Protective Equipment (PPE) compliance** on construction sites. It leverages the state-of-the-art **YOLOv12** object detection model to analyze images, identify workers, and instantly report on their adherence to safety standards (helmets and vests).

This application was developed as a final capstone project for the **Purwadhika AI Engineering** course.

---

## ✨ Key Features

* **State-of-the-Art Detection:** Built on **YOLOv12**, fine-tuned on a custom construction safety dataset for high accuracy.
* **KPI Dashboard:** Automatically generates a Key Performance Indicator (KPI) dashboard for each image, summarizing:
    * Total Workers Detected
    * Total Compliance Violations
    * Overall Compliance Rate (%)
* **Multi-Image Auditing:** Users can upload multiple images at once for batch processing.
* **Severity-Based Reporting:** Classifies workers into three color-coded categories for quick visual analysis:
    * 🟩 **AMAN (Safe):** Helmet and vest detected.
    * 🟨 **PERINGATAN (Warning):** Missing vest.
    * 🟥 **BAHAYA (Danger):** Missing helmet.
* **Toggleable Privacy Blurring:** Includes an optional "Privacy Blur" feature that automatically blurs the head region of detected workers, which can be toggled on or off in the sidebar.
* **Detailed Reporting:** Provides a per-worker breakdown of compliance status in a clear, tabular format.

---

## 🛠️ Tech Stack

* **Model:** YOLOv12 (from the [official repository](https://github.com/sunsmarterjie/yolov12))
* **App Framework:** Streamlit
* **CV & Data:** OpenCV, Pandas, Pillow (PIL), NumPy
* **Training:** Google Colab (documented in the Jupyter Notebook)

---

## 📙 Model Training & Analysis

The complete process of data exploration, evaluation, and fine-tuning the YOLOv12 model is documented in the Jupyter Notebook:
`capstone4_thariq_ahmad_ppe_audit.ipynb`

This notebook includes:
* Environment setup for YOLOv12.
* Live training monitoring using TensorBoard.
* In-depth analysis of model performance metrics (mAP, Precision-Recall curves, F1-Score, and Loss).
* Prototyping of the `run_ppe_audit` post-processing logic.

---

## 🏃‍♂️ How to Run Locally

### 1. Prerequisites

* Python 3.10 or newer
* Git

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd capstone-ppe-audit
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This `requirements.txt` file will install Streamlit, OpenCV, and the specific YOLOv12 repo from GitHub.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download/Place the Model:**
    * Download the fine-tuned model weights (`yolov12s_best_v2.pt`).
    * Place them inside the `models/` folder. (The app expects `models/yolov12s_best_v2.pt`).

### 3. Run the App

1.  **Launch Streamlit:**
    ```bash
    streamlit run app.py
    ```
2.  Open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 🚀 Future Improvements

This project serves as a strong foundation. Future planned updates include:

* [ ] **Video & Live Stream Support:** Adapt the auditing logic to work with video files and real-time RTSP/webcam feeds.
* [ ] **Improve Class-Specific Accuracy:** Collect more data for the `no-helmet` and `no-vest` classes to improve model robustness (addressing the class imbalance found during analysis).
* [ ] **Automated Alerting:** Integrate an alerting system (e.g., email or SMS) to notify supervisors of critical violations in real-time.
