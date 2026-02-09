import streamlit as st
import cv2
import torch
import math
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Nagrak Traffic Intelligence",
    page_icon="🚗",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Metric Container Gelap */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 28, 28, 0.5);
        border: 1px solid #444;
        padding: 10px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="metric-container"] label {
        color: #aaaaaa !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Box Total Kendaraan */
    .total-box {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444;
        margin-bottom: 20px;
    }
    .total-number {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
    }
    .total-label {
        font-size: 16px;
        color: #aaaaaa;
        margin: 0;
    }
    
    /* Box Insight Report */
    .insight-box {
        background-color: #1E1E1E;
        border-left: 5px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        color: #E0E0E0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Mengatur tinggi Peta */
    iframe[title="streamlit_folium.st_folium"] {
        height: 200px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TRACKER CLASS ---
class SimpleTracker:
    def __init__(self, distance_threshold=80):
        self.stored_data = {}
        self.id_count = 0
        self.distance_threshold = distance_threshold

    def update(self, objects_rect_with_labels):
        objects_bbs_ids_labels = []
        for rect in objects_rect_with_labels:
            x, y, w, h, label = rect
            cx, cy = (x + x + w) // 2, (y + y + h) // 2
            same_object = False
            for id, data in self.stored_data.items():
                center_pt = data[0]
                dist = math.hypot(cx - center_pt[0], cy - center_pt[1])
                if dist < self.distance_threshold:
                    self.stored_data[id] = ((cx, cy), data[1])
                    objects_bbs_ids_labels.append([x, y, w, h, id, data[1]])
                    same_object = True
                    break
            if not same_object:
                self.stored_data[self.id_count] = ((cx, cy), label)
                objects_bbs_ids_labels.append([x, y, w, h, self.id_count, label])
                self.id_count += 1
        new_stored_data = {}
        for obj in objects_bbs_ids_labels:
            new_stored_data[obj[4]] = self.stored_data[obj[4]]
        self.stored_data = new_stored_data.copy()
        return objects_bbs_ids_labels

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.35
    model.classes = [2, 5, 7]  # Car, Bus, Truck
    return model

# --- MAIN APP ---
def main():
    # Title and Description
    st.markdown("<h1 style='text-align: center;'>🚗 Nagrak Traffic Intelligence System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Real-time analysis of vehicle traffic flow, classification, and directional tracking.</p>", unsafe_allow_html=True)
    st.write("") # Spacer

    # Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, 'data', 'input', 'ext_tol_nagrak_night.mp4')

    if not os.path.exists(video_path):
        st.error(f"Error: Video file not found at {video_path}")
        return

    # Grid Layout
    col_video, col_stats = st.columns([2, 1])

    with col_video:
        st.subheader("📹 Real-time CCTV Feed")
        stframe = st.empty()
        
        # --- Insight Report ---
        st.write("") 
        st.subheader("💡Insight Report")
        insight_placeholder = st.empty()

    with col_stats:
        st.subheader("📊 Traffic Statistics")
        
        # --- MAPS ---
        st.caption("📍GT Nagrak (Kota Wisata, Kabupaten Bogor)")
        map_data = pd.DataFrame({'lat': [-6.3839], 'lon': [106.9455]})
        st.map(map_data, zoom=14, color='#FF0000', size=40, use_container_width=True)
        
        st.write("") 
        
        # 1. Total Vehicles Box
        kpi_total_placeholder = st.empty()
        
        # 2. Charts
        st.write("###### 1. Direction Distribution")
        chart_direction = st.empty()
        
        st.write("###### 2. Vehicle Classification")
        chart_vehicle = st.empty()

    # Initialization
    model = load_model()
    tracker = SimpleTracker()
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    line_y = int(height * 0.45)

    count_out = 0
    count_in = 0
    stats_vehicle = {"Car": 0, "Bus": 0, "Truck": 0}
    previous_positions = {}
    
    # Simulation Start Time
    start_time = datetime(2026, 2, 6, 20, 34, 0)
    
    frame_count = 0
    
    # Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue

        frame_count += 1
        
        # Detection
        results = model(frame)
        df = results.pandas().xyxy[0]
        detections = []
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name'].capitalize()
            detections.append([x1, y1, x2-x1, y2-y1, label])

        # Tracking
        track_results = tracker.update(detections)

        # Visuals
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        
        for res in track_results:
            x, y, w, h, id, label = res
            cx, cy = (x + w//2), (y + h//2)
            
            if id in previous_positions:
                prev_y = previous_positions[id]
                if prev_y < line_y and cy >= line_y:
                    count_out += 1
                    stats_vehicle[label] += 1
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 5)
                elif prev_y > line_y and cy <= line_y:
                    count_in += 1
                    stats_vehicle[label] += 1
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 5)

            previous_positions[id] = cy
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, f"{id}|{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # Update Metrics
        total_vehicle = count_out + count_in
        
        kpi_total_placeholder.markdown(f"""
        <div class="total-box">
            <p class="total-number">{total_vehicle}</p>
            <p class="total-label">Total Vehicles Detected</p>
        </div>
        """, unsafe_allow_html=True)

        df_dir = pd.DataFrame({"Direction": ["OUT (Exit)", "IN (Entry)"], "Count": [count_out, count_in]})
        chart_direction.bar_chart(df_dir.set_index("Direction"), color=["#FF4B4B"])

        df_veh = pd.DataFrame({"Type": list(stats_vehicle.keys()), "Count": list(stats_vehicle.values())})
        chart_vehicle.bar_chart(df_veh.set_index("Type"), color=["#00CC96"])

        # Narrative Logic
        elapsed_seconds = frame_count / fps
        current_time = start_time + timedelta(seconds=elapsed_seconds)
        duration_minutes = elapsed_seconds / 60
        
        date_str = start_time.strftime("%d %B %Y")
        time_start_str = start_time.strftime("%H:%M")
        time_curr_str = current_time.strftime("%H:%M")
        
        report_text = f"""
        Dalam durasi <b>{duration_minutes:.2f} menit</b> pada tanggal <b>{date_str}</b> 
        pukul <b>{time_start_str}-{time_curr_str}</b>, pada exit tol ini terpantau:
        <br><br>
        • <b>{count_out}</b> kendaraan <b>Keluar Tol</b><br>
        • <b>{count_in}</b> kendaraan <b>Masuk Tol</b><br>
        <br>
        Komposisi kendaraan terdiri dari:<br>
        🚗 <b>{stats_vehicle['Car']}</b> Mobil<br>
        🚌 <b>{stats_vehicle['Bus']}</b> Bus<br>
        🚛 <b>{stats_vehicle['Truck']}</b> Truk
        """
        
        insight_placeholder.markdown(f"<div class='insight-box'>{report_text}</div>", unsafe_allow_html=True)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

if __name__ == '__main__':
    main()