import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from ultralytics import YOLO  # Ini akan mengimpor dari repo git yang kita instal

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(
    page_title="Audit K3 (PPE) Otomatis",
    page_icon="👷‍♂️",
    layout="wide"
)

st.title("👷‍♂️ Audit K3 (PPE) Otomatis")
st.write("Unggah gambar pekerja konstruksi untuk menganalisis kepatuhan APD (Helm & Rompi) secara otomatis.")

# --- Fungsi Helper (dari Bab 5) ---

@st.cache_data
def calculate_iou(box1, box2):
    """Menghitung Intersection over Union (IoU)"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

def run_ppe_audit(model_results, image_bgr, confidence_threshold=0.5, blur_heads=True):
    """
    Memproses hasil deteksi model dan melakukan audit K3.
    (Versi Final dengan Toggle Privacy Blurring)
    """
    annotated_image = image_bgr.copy()
    boxes_data = model_results[0].boxes

    persons, helmets, vests, no_helmets, no_vests = [], [], [], [], []

    for box in boxes_data:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        
        if conf < confidence_threshold:
            continue

        class_name = model.names[class_id]

        if class_name == 'person':
            persons.append(coords)
            if blur_heads:
                (x1, y1, x2, y2) = coords
                person_height = y2 - y1
                head_height = int(person_height * 0.25)
                head_y1, head_y2 = y1, y1 + head_height
                head_y1, head_y2 = max(0, head_y1), min(annotated_image.shape[0], head_y2)
                x1, x2 = max(0, x1), min(annotated_image.shape[1], x2)
                
                if head_y1 < head_y2 and x1 < x2:
                    head_roi = annotated_image[head_y1:head_y2, x1:x2] 
                    blurred_roi = cv2.GaussianBlur(head_roi, (51, 51), 0)
                    annotated_image[head_y1:head_y2, x1:x2] = blurred_roi
        
        elif class_name == 'helmet': helmets.append(coords)
        elif class_name == 'vest': vests.append(coords)
        elif class_name == 'no-helmet': no_helmets.append(coords)
        elif class_name == 'no-vest': no_vests.append(coords)

    compliance_report = []
    for person_id, p_box in enumerate(persons, 1):
        status = {"id": f"Pekerja {person_id}", "has_helmet": False, "has_vest": False, 
                  "no_helmet_detected": False, "no_vest_detected": False, "box": p_box}
        for h_box in helmets:
            if calculate_iou(p_box, h_box) > 0.1: status["has_helmet"] = True; break
        for v_box in vests:
            if calculate_iou(p_box, v_box) > 0.2: status["has_vest"] = True; break
        for nh_box in no_helmets:
            if calculate_iou(p_box, nh_box) > 0.1: status["no_helmet_detected"] = True
        for nv_box in no_vests:
            if calculate_iou(p_box, nv_box) > 0.1: status["no_vest_detected"] = True
        compliance_report.append(status)

    for report in compliance_report:
        p_box = report["box"]
        (x1, y1, x2, y2) = p_box
        is_critical = not report["has_helmet"] or report["no_helmet_detected"]
        is_warning = not report["has_vest"] or report["no_vest_detected"]

        if is_critical:
            status_text, color = "BAHAYA (Helm)", (0, 0, 255) # Merah
        elif is_warning and not is_critical: 
            status_text, color = "PERINGATAN (Rompi)", (0, 255, 255) # Kuning
        else:
            status_text, color = "AMAN", (0, 255, 0) # Hijau

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, f"{report['id']}: {status_text}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return compliance_report, annotated_image

# --- Memuat Model (dengan Cache) ---
@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dan menyimpannya di cache Streamlit."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

model_path = "models/yolov12s_best_v2.pt"
model = load_model(model_path)

# --- Sidebar (Input Pengguna) ---
st.sidebar.header("⚙️ Pengaturan Audit")
confidence = st.sidebar.slider(
    "Ambang Batas Confidence", 
    min_value=0.1, max_value=1.0, value=0.4, step=0.05
)
blur_toggle = st.sidebar.checkbox(
    "Aktifkan Privacy Blur (Kepala)", 
    value=True
)

st.sidebar.header("📁 Upload Gambar")
uploaded_files = st.sidebar.file_uploader(
    "Upload satu atau beberapa gambar...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # <-- KUNCI UNTUK MULTI-UPLOAD
)

# --- Area Tampilan Utama ---
if not model:
    st.error("Model gagal dimuat. Aplikasi tidak dapat berjalan.")
elif not uploaded_files:
    st.info("Silakan upload gambar melalui sidebar untuk memulai audit.")
else:
    st.success(f"Berhasil memuat {len(uploaded_files)} gambar. Memulai proses audit...")

    # Loop untuk setiap file yang di-upload
    for uploaded_file in uploaded_files:
        st.divider() # Garis pemisah
        st.header(f"Hasil Audit untuk: `{uploaded_file.name}`")
        
        # 1. Konversi file upload ke format OpenCV
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 2. Jalankan prediksi dan audit
        results = model.predict(image_bgr, verbose=False)
        laporan, img_hasil = run_ppe_audit(
            results, 
            image_bgr, 
            confidence_threshold=float(confidence), 
            blur_heads=blur_toggle
        )

        # 3. --- FITUR BARU: KPI Dashboard ---
        total_pekerja = len(laporan)
        total_pelanggaran = 0
        total_aman = 0

        for item in laporan:
            is_safe = item["has_helmet"] and item["has_vest"]
            if not is_safe:
                total_pelanggaran += 1
            else:
                total_aman += 1
        
        kepatuhan_percent = (total_aman / total_pekerja) * 100 if total_pekerja > 0 else 100

        st.subheader("📊 Ringkasan Kepatuhan (KPI)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pekerja", f"{total_pekerja} 👷‍♂️")
        col2.metric("Total Pelanggaran", f"{total_pelanggaran} ❌")
        col3.metric("Tingkat Kepatuhan", f"{kepatuhan_percent:.1f} %")
        
        if total_pelanggaran > 0:
            st.error("Ditemukan pelanggaran K3! Periksa laporan rinci di bawah.")
        else:
            st.success("Semua pekerja yang terdeteksi telah patuh.")

        # 4. Tampilkan Gambar Hasil
        st.subheader("🖼️ Hasil Audit Visual")
        image_rgb = cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB) # Konversi BGR ke RGB untuk Streamlit
        st.image(image_rgb, caption="Gambar Hasil Audit (dengan anotasi)")

        # 5. Tampilkan Laporan Rinci
        st.subheader("📋 Laporan Rinci Per Pekerja")
        if laporan:
            # Ubah list of dicts menjadi DataFrame agar rapi
            df_laporan = pd.DataFrame(laporan)
            
            # Buat kolom status yang mudah dibaca
            def get_helm_status(row):
                if row['has_helmet']: return "✅ Patuh"
                if row['no_helmet_detected']: return "❌ Pelanggaran (Terdeteksi)"
                return "❌ Pelanggaran (Tidak Ada)"

            def get_vest_status(row):
                if row['has_vest']: return "✅ Patuh"
                if row['no_vest_detected']: return "❌ Pelanggaran (Terdeteksi)"
                return "❌ Pelanggaran (Tidak Ada)"

            df_laporan['Status Helm'] = df_laporan.apply(get_helm_status, axis=1)
            df_laporan['Status Rompi'] = df_laporan.apply(get_vest_status, axis=1)
            
            # Tampilkan tabel
            st.dataframe(
                df_laporan[['id', 'Status Helm', 'Status Rompi']],
                use_container_width=True
            )
        else:
            st.info("Tidak ada pekerja yang terdeteksi di gambar ini.")