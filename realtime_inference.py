"""
Real-Time Driver Drowsiness Detection & Vigilance Monitoring System
DyG-STMT v4 / v3 End-to-End Pipeline
-------------------------------------------------------------------
Features:
- High-Throughput Spatial CNN Encoders (Eye & Mouth 64x64 ROIs -> 128-D)
- SOTA DyG-STMT Sequence Model with SKCM Cross-Modulation & T-SE
- Dynamic Multi-Modal Fusion (Deep Features + Kinematic PERCLOS + Adaptive EAR)
- Interactive HUD Overlay with Mini-PIP Crops, Score Meter, & Scenario Switcher
- Audio/Visual Wake-Up Alarm on Prolonged Eye Closure or High Drowsiness
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Optional audio alert on Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from dataset.feature_extractor import FacialFeatureExtractor
from models.dyg_stmt import FastSpatialEncoder
from models.dyg_stmt_v4 import DyGSTMTv4
from models.dyg_stmt_v3 import DyGSTMTv3
from models.dyg_stmt_v2 import DyGSTMTv2

CONDITIONS = [
    "No Glasses (Day)",
    "Regular Glasses (Day)",
    "Sunglasses (Day)",
    "Night (Bare Face / IR)",
    "Night (With Glasses / IR)"
]

def load_sequence_model(checkpoint_path, device):
    """
    Automatically detects and loads the appropriate DyG-STMT architecture version.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    
    # Check architecture signatures
    keys = list(state_dict.keys())
    if any('skcm_fusion' in k for k in keys) or any('tri_tcn' in k for k in keys):
        print("[Model Loader] Detected Architecture: DyG-STMT v4 (Max-Vigilance Ultra SOTA)")
        model = DyGSTMTv4().to(device)
    elif any('bcma_fusion' in k for k in keys):
        print("[Model Loader] Detected Architecture: DyG-STMT v3 (Bi-Cross-Modal SOTA)")
        model = DyGSTMTv3().to(device)
    elif any('ms_d_tcn' in k for k in keys):
        print("[Model Loader] Detected Architecture: DyG-STMT v2 (Enhanced)")
        model = DyGSTMTv2().to(device)
    else:
        print("[Model Loader] Defaulting to DyG-STMT v4 architecture.")
        model = DyGSTMTv4().to(device)
        
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model Loader] Successfully loaded model weights from: {checkpoint_path}")
    return model

def load_spatial_encoders(device, eye_ckpt='checkpoints/pretrained_eye_encoder.pt', mouth_ckpt='checkpoints/pretrained_mouth_encoder.pt'):
    """
    Loads pretrained FastSpatialEncoders for 64x64 Eye and Mouth ROIs.
    """
    eye_encoder = FastSpatialEncoder(out_dim=128).to(device)
    mouth_encoder = FastSpatialEncoder(out_dim=128).to(device)
    
    if os.path.exists(eye_ckpt):
        eye_encoder.load_state_dict(torch.load(eye_ckpt, map_location=device))
        print(f"[Encoder Loader] Loaded Eye Spatial Encoder: {eye_ckpt}")
    else:
        print(f"[Encoder Loader] Warning: '{eye_ckpt}' not found. Using initialized weights.")
        
    if os.path.exists(mouth_ckpt):
        mouth_encoder.load_state_dict(torch.load(mouth_ckpt, map_location=device))
        print(f"[Encoder Loader] Loaded Mouth Spatial Encoder: {mouth_ckpt}")
    else:
        print(f"[Encoder Loader] Warning: '{mouth_ckpt}' not found. Using initialized weights.")
        
    eye_encoder.eval()
    mouth_encoder.eval()
    return eye_encoder, mouth_encoder

def draw_hud(frame, alert_level, alert_color, drowsy_score, fps, ear, mar, perclos, head_pose, cond_name, eye_crop, mouth_crop, flash_alert=False):
    """
    Renders a modern, high-contrast HUD on the camera stream.
    """
    h, w = frame.shape[:2]
    
    # 1. Flash border on critical drowsiness alarm
    if flash_alert and (int(time.time() * 4) % 2 == 0):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        
    # 2. Main Status Card (Top Left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (420, 210), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (420, 210), alert_color, 2)
    
    # Title & State
    cv2.putText(frame, "DyG-STMT MONITOR v4", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, f"{alert_level}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, alert_color, 2)
    
    # Drowsiness Score Bar
    cv2.putText(frame, f"Drowsiness Score: {drowsy_score:.1f}%", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    bar_x, bar_y, bar_w, bar_h = 30, 120, 360, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * np.clip(drowsy_score / 100.0, 0.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), alert_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    
    # Telemetry Line 1: EAR / MAR / PERCLOS
    cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f} | PERCLOS: {perclos*100:.0f}%", 
                (30, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                
    # Telemetry Line 2: FPS & Condition
    pitch, yaw, roll = head_pose
    cv2.putText(frame, f"FPS: {fps:.1f} | Pose: P:{pitch:.0f} Y:{yaw:.0f} | [C]: {cond_name[:14]}", 
                (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    # 3. Mini-PIP ROIs (Top Right)
    pip_w, pip_h = 80, 80
    if eye_crop is not None and mouth_crop is not None:
        e_pip = (cv2.resize(eye_crop, (pip_w, pip_h)) * 255).astype(np.uint8)
        m_pip = (cv2.resize(mouth_crop, (pip_w, pip_h)) * 255).astype(np.uint8)
        e_pip = cv2.cvtColor(e_pip, cv2.COLOR_RGB2BGR)
        m_pip = cv2.cvtColor(m_pip, cv2.COLOR_RGB2BGR)
        
        # Eye PIP
        ex1, ey1 = w - pip_w - 20, 20
        frame[ey1:ey1+pip_h, ex1:ex1+pip_w] = e_pip
        cv2.rectangle(frame, (ex1, ey1), (ex1+pip_w, ey1+pip_h), (0, 255, 255), 1)
        cv2.putText(frame, "EYE ROI", (ex1 + 5, ey1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        
        # Mouth PIP
        mx1, my1 = w - (pip_w * 2) - 35, 20
        frame[my1:my1+pip_h, mx1:mx1+pip_w] = m_pip
        cv2.rectangle(frame, (mx1, my1), (mx1+pip_w, my1+pip_h), (255, 180, 0), 1)
        cv2.putText(frame, "MOUTH ROI", (mx1 + 5, my1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 0), 1)

    # 4. Footer controls
    cv2.putText(frame, "Keys: [C] Toggle Condition | [Q] Quit", (20, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)


def run_realtime_inference(checkpoint_path='checkpoints/best_dyg_stmt_v4.pt', video_source=0, initial_cond=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================")
    print(f"  DyG-STMT REAL-TIME DRIVER MONITORING PIPELINE  ")
    print(f"==================================================")
    print(f"[Device] Compute target: {device}")
    
    # 1. Load Model & Spatial Encoders
    model = load_sequence_model(checkpoint_path, device)
    eye_encoder, mouth_encoder = load_spatial_encoders(device)
    extractor = FacialFeatureExtractor(roi_size=(64, 64))
    
    # 2. Setup Video Capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{video_source}'. Please check your webcam connection.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[Controls] Starting video feed...")
    print(" - Press 'C' in the window to cycle driving scenario (Bareface, Glasses, Sunglasses, Night).")
    print(" - Press 'Q' to exit.")
    
    # Rolling feature buffers for T=30 window
    buffer_eye_feat = []
    buffer_mouth_feat = []
    buffer_geo = []
    
    # Current condition index
    current_cond = initial_cond
    
    # Telemetry tracking
    fps_time = time.time()
    frame_counter = 0
    fps = 0.0
    
    drowsy_score_smooth = 15.0
    alert_level = "ALERT"
    alert_color = (0, 255, 0)
    last_alarm_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Stream] Video stream ended or camera disconnected.")
            break
            
        frame_counter += 1
        curr_time = time.time()
        if curr_time - fps_time >= 0.5:
            fps = frame_counter / (curr_time - fps_time)
            fps_time = curr_time
            frame_counter = 0
            
        # 1. Extract Face Landmarks, Aspect Ratios & ROIs
        cond_name = CONDITIONS[current_cond]
        eye_roi, mouth_roi, geo_vec = extractor.process_frame(frame, condition_name=cond_name)
        
        # 2. Extract Spatial CNN Embeddings (64x64 -> 128-D)
        t_eye_crop = torch.from_numpy(np.transpose(eye_roi, (2, 0, 1))).unsqueeze(0).float().to(device)
        t_mouth_crop = torch.from_numpy(np.transpose(mouth_roi, (2, 0, 1))).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            eye_emb = eye_encoder(t_eye_crop).cpu().numpy()[0]     # (128,)
            mouth_emb = mouth_encoder(t_mouth_crop).cpu().numpy()[0] # (128,)
            
        # 3. Append to Rolling T=30 Buffers
        buffer_eye_feat.append(eye_emb)
        buffer_mouth_feat.append(mouth_emb)
        buffer_geo.append(geo_vec)
        
        if len(buffer_eye_feat) > 30:
            buffer_eye_feat.pop(0)
            buffer_mouth_feat.pop(0)
            buffer_geo.pop(0)
            
        # 4. Run Sequence Model Forward Pass
        if len(buffer_eye_feat) == 30:
            t_eye = torch.from_numpy(np.array([buffer_eye_feat], dtype=np.float32)).to(device)
            t_mouth = torch.from_numpy(np.array([buffer_mouth_feat], dtype=np.float32)).to(device)
            t_geo = torch.from_numpy(np.array([buffer_geo], dtype=np.float32)).to(device)
            
            cond_tensor = torch.zeros(1, 5, dtype=torch.float32).to(device)
            cond_tensor[0, current_cond] = 1.0
            
            with torch.no_grad():
                out = model(t_eye, t_mouth, t_geo, cond_tensor)
                raw_prob = torch.sigmoid(out['drowsy_logits']).item()
                
            # Kinetic PERCLOS & EAR Safety Fusion
            perclos_val = float(geo_vec[11])
            ear_val = float(geo_vec[0]) # Left EAR
            mar_val = float(geo_vec[3]) # MAR
            
            # If eyes are closed (EAR < 0.20 or PERCLOS > 0.40), fuse dynamic penalty
            eye_closure_factor = 0.0
            if ear_val < 0.20:
                eye_closure_factor = min(1.0, (0.20 - ear_val) / 0.12)
            if perclos_val > 0.35:
                eye_closure_factor = max(eye_closure_factor, perclos_val)
                
            combined_prob = np.clip(0.70 * raw_prob + 0.30 * eye_closure_factor, 0.0, 1.0)
            raw_score = combined_prob * 100.0
            
            # Smooth EMA to avoid single-frame flickering
            drowsy_score_smooth = 0.75 * drowsy_score_smooth + 0.25 * raw_score
        else:
            # Filling buffer indicator
            perclos_val = float(geo_vec[11])
            ear_val = float(geo_vec[0])
            mar_val = float(geo_vec[3])
            drowsy_score_smooth = 10.0
            
        # 5. State Classification & Warning Triggers
        is_flashing = False
        if drowsy_score_smooth < 35.0:
            alert_level = "ALERT (VIGILANT)"
            alert_color = (0, 255, 0) # Green
        elif drowsy_score_smooth < 65.0:
            alert_level = "LOW VIGILANCE"
            alert_color = (0, 165, 255) # Orange
        else:
            alert_level = "DROWSY - WAKE UP!"
            alert_color = (0, 0, 255) # Red
            is_flashing = True
            
            # Trigger audio beep on critical alarm (rate-limited to 1 per 0.8s)
            if HAS_WINSOUND and (curr_time - last_alarm_time >= 0.8):
                try:
                    winsound.Beep(1400, 250)
                    last_alarm_time = curr_time
                except Exception:
                    pass
                    
        # 6. Render HUD
        head_pose = (geo_vec[5] * 90.0, geo_vec[6] * 90.0, geo_vec[7] * 90.0)
        draw_hud(
            frame,
            alert_level=alert_level,
            alert_color=alert_color,
            drowsy_score=drowsy_score_smooth,
            fps=fps,
            ear=ear_val,
            mar=mar_val,
            perclos=perclos_val,
            head_pose=head_pose,
            cond_name=CONDITIONS[current_cond],
            eye_crop=eye_roi,
            mouth_crop=mouth_roi,
            flash_alert=is_flashing
        )
        
        cv2.imshow("DyG-STMT Real-Time Driver Drowsiness Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[Inference] User requested exit.")
            break
        elif key == ord('c'):
            current_cond = (current_cond + 1) % len(CONDITIONS)
            print(f"[Condition Switch] Switched to scenario: {CONDITIONS[current_cond]}")
            
    cap.release()
    cv2.destroyAllWindows()
    print("[Inference] Application closed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DyG-STMT Real-Time Driver Drowsiness Inference")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_dyg_stmt_v4.pt',
                        help='Path to DyG-STMT checkpoint (.pt)')
    parser.add_argument('--source', type=str, default='0',
                        help='Camera index (e.g. 0) or video file path')
    parser.add_argument('--condition', type=int, default=0,
                        help='Initial driving condition index (0: NoGlasses, 1: Glasses, 2: Sunglasses, 3: Night, 4: NightGlasses)')
    args = parser.parse_args()
    
    src = int(args.source) if args.source.isdigit() else args.source
    run_realtime_inference(
        checkpoint_path=args.checkpoint,
        video_source=src,
        initial_cond=args.condition
    )
