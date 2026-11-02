import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.nthu_data_loader import parse_label_file, CONDITION_MAP
from dataset.feature_extractor import FacialFeatureExtractor
import cv2

def extract_and_save_video_shard(args_tuple):
    try:
        vid_path, d_label, e_label, m_label, h_label, cond, num_windows, out_shard_path = args_tuple
        
        if os.path.exists(out_shard_path):
            return out_shard_path
            
        if not os.path.exists(d_label):
            return None
            
        labels_drowsy = parse_label_file(d_label)
        if labels_drowsy is None or len(labels_drowsy) == 0:
            return None
            
        labels_eye = parse_label_file(e_label) if e_label and os.path.exists(e_label) else None
        labels_mouth = parse_label_file(m_label) if m_label and os.path.exists(m_label) else None
        labels_head = parse_label_file(h_label) if h_label and os.path.exists(h_label) else None
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seq_len = 30
        valid_len = min(total_frames, len(labels_drowsy))
        
        if valid_len < seq_len + 1:
            cap.release()
            return None
            
        n_win = min(num_windows, max(1, valid_len // seq_len))
        start_indices = sorted(list(set(np.linspace(0, valid_len - seq_len - 1, n_win, dtype=int))))
        
        needed_frames = {}
        for s_idx, s in enumerate(start_indices):
            for f in range(s, s + seq_len):
                needed_frames[f] = (s_idx, f - s)
                
        extractor = FacialFeatureExtractor(roi_size=(64, 64))
        cond_idx = CONDITION_MAP.get(cond.lower(), 0)
        
        # Preallocate windows
        n_actual_win = len(start_indices)
        seq_eye = np.zeros((n_actual_win, seq_len, 3, 64, 64), dtype=np.uint8)
        seq_mouth = np.zeros((n_actual_win, seq_len, 3, 64, 64), dtype=np.uint8)
        seq_geo = np.zeros((n_actual_win, seq_len, 16), dtype=np.float32)
        
        max_needed_frame = max(needed_frames.keys())
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > max_needed_frame:
                break
                
            if frame_idx in needed_frames:
                w_idx, offset = needed_frames[frame_idx]
                eye_roi, mouth_roi, geo_vec = extractor.process_frame(frame, condition_name=cond)
                
                eye_uint8 = (eye_roi * 255.0).clip(0, 255).astype(np.uint8)
                mouth_uint8 = (mouth_roi * 255.0).clip(0, 255).astype(np.uint8)
                
                seq_eye[w_idx, offset] = np.transpose(eye_uint8, (2, 0, 1))
                seq_mouth[w_idx, offset] = np.transpose(mouth_uint8, (2, 0, 1))
                seq_geo[w_idx, offset] = geo_vec
                
            frame_idx += 1
            
        cap.release()
        
        labels_eye = labels_eye[:valid_len] if labels_eye is not None else np.zeros(valid_len, dtype=np.int64)
        labels_mouth = labels_mouth[:valid_len] if labels_mouth is not None else np.zeros(valid_len, dtype=np.int64)
        labels_head = labels_head[:valid_len] if labels_head is not None else np.zeros(valid_len, dtype=np.int64)
        
        seq_drowsy = []
        seq_l_eye = []
        seq_l_mouth = []
        seq_l_head = []
        
        for s in start_indices:
            end = s + seq_len
            w_drowsy = labels_drowsy[s:end]
            w_label = 1 if np.mean(w_drowsy) >= 0.5 else 0
            seq_drowsy.append(w_label)
            seq_l_eye.append(labels_eye[s:end])
            seq_l_mouth.append(labels_mouth[s:end])
            seq_l_head.append(labels_head[s:end])
            
        os.makedirs(os.path.dirname(out_shard_path), exist_ok=True)
        np.savez_compressed(
            out_shard_path,
            eye_imgs=seq_eye,
            mouth_imgs=seq_mouth,
            geo_vecs=seq_geo,
            cond_idx=np.array([cond_idx] * n_actual_win, dtype=np.int64),
            label_drowsy=np.array(seq_drowsy, dtype=np.int64),
            label_eye=np.array(seq_l_eye, dtype=np.float32),
            label_mouth=np.array(seq_l_mouth, dtype=np.float32),
            label_head=np.array(seq_l_head, dtype=np.float32)
        )
        return out_shard_path
    except Exception as e:
        print(f"[Warning] Error extracting shard: {e}")
        return None


def run_parallel_tasks(tasks, desc="Extracting Shards", max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_and_save_video_shard, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                f.result()
            except Exception as e:
                pass


def preprocess_all_splits(raw_dir='raw', output_dir='data_cache', train_windows=20, eval_windows=35, test_windows=35, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Train Tasks
    train_path = os.path.join(raw_dir, 'nthu_training')
    train_shards_dir = os.path.join(output_dir, 'train_shards')
    train_tasks = []
    for subj in sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]):
        subj_dir = os.path.join(train_path, subj)
        for cond in [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]:
            cond_dir = os.path.join(subj_dir, cond)
            for avi_path in glob.glob(os.path.join(cond_dir, "*.avi")):
                b_name = os.path.splitext(os.path.basename(avi_path))[0]
                d_label = os.path.join(cond_dir, f"{subj}_{b_name}_drowsiness.txt")
                e_label = os.path.join(cond_dir, f"{subj}_{b_name}_eye.txt")
                m_label = os.path.join(cond_dir, f"{subj}_{b_name}_mouth.txt")
                h_label = os.path.join(cond_dir, f"{subj}_{b_name}_head.txt")
                out_shard = os.path.join(train_shards_dir, f"{subj}_{cond}_{b_name}.npz")
                train_tasks.append((avi_path, d_label, e_label, m_label, h_label, cond, train_windows, out_shard))
                
    # 2. Eval Tasks
    eval_path = os.path.join(raw_dir, 'nthu_evaluation')
    eval_shards_dir = os.path.join(output_dir, 'eval_shards')
    eval_tasks = []
    for subj in sorted([d for d in os.listdir(eval_path) if os.path.isdir(os.path.join(eval_path, d))]):
        subj_dir = os.path.join(eval_path, subj)
        for mp4 in glob.glob(os.path.join(subj_dir, "*.mp4")):
            base = os.path.splitext(os.path.basename(mp4))[0]
            cond_key = base.replace(f"{subj}_", "").replace("_mix", "")
            d_label = os.path.join(subj_dir, f"{subj}_{cond_key}_mixing_drowsiness.txt")
            if not os.path.exists(d_label):
                d_label = os.path.join(subj_dir, f"{base}_drowsiness.txt")
            out_shard = os.path.join(eval_shards_dir, f"{subj}_{cond_key}.npz")
            eval_tasks.append((mp4, d_label, None, None, None, cond_key, eval_windows, out_shard))
            
    # 3. Test Tasks
    test_path = os.path.join(raw_dir, 'nthu_testing')
    test_shards_dir = os.path.join(output_dir, 'test_shards')
    label_base = os.path.join(test_path, 'test_label_txt')
    test_tasks = []
    for mp4 in glob.glob(os.path.join(test_path, "*.mp4")):
        fname = os.path.basename(mp4)
        parts = fname.replace('.mp4', '').split('_')
        if len(parts) >= 2:
            subj = parts[0]
            cond_key = parts[1]
            cond_dir_name = "night_noglasses" if cond_key == "nightnoglasses" else ("night_glasses" if cond_key == "nightglasses" else cond_key)
            d_label = os.path.join(label_base, cond_dir_name, f"{subj}_{cond_key}_mixing_drowsiness.txt")
            if not os.path.exists(d_label):
                cands = glob.glob(os.path.join(label_base, "*", f"{subj}_{cond_key}*.txt"))
                if cands:
                    d_label = cands[0]
            out_shard = os.path.join(test_shards_dir, f"{subj}_{cond_key}.npz")
            test_tasks.append((mp4, d_label, None, None, None, cond_key, test_windows, out_shard))

    print(f"\n[Parallel Full-Video Window Extraction]: {len(train_tasks)} train, {len(eval_tasks)} eval, {len(test_tasks)} test videos\n")
    
    run_parallel_tasks(train_tasks, desc="Extracting Train Shards", max_workers=max_workers)
    run_parallel_tasks(eval_tasks, desc="Extracting Eval Shards", max_workers=max_workers)
    run_parallel_tasks(test_tasks, desc="Extracting Test Shards", max_workers=max_workers)
    
    print("\nAll shards successfully extracted and cached to disk!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='raw')
    parser.add_argument('--output_dir', type=str, default='data_cache')
    parser.add_argument('--train_windows', type=int, default=20)
    parser.add_argument('--eval_windows', type=int, default=35)
    parser.add_argument('--test_windows', type=int, default=35)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    preprocess_all_splits(
        args.raw_dir, 
        args.output_dir, 
        train_windows=args.train_windows, 
        eval_windows=args.eval_windows, 
        test_windows=args.test_windows, 
        max_workers=args.workers
    )
