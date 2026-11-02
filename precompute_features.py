import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.dyg_stmt import FastSpatialEncoder

def extract_features_for_dataset(npz_input_path, npz_output_path, batch_size=64):
    print(f"\n[Precompute Features] Reading {npz_input_path}...")
    data = np.load(npz_input_path, mmap_mode='r')
    
    eye_imgs = data['eye_imgs']
    mouth_imgs = data['mouth_imgs']
    geo_vecs = data['geo_vecs']
    cond_idx = data['cond_idx']
    label_drowsy = data['label_drowsy']
    label_eye = data['label_eye']
    label_mouth = data['label_mouth']
    label_head = data['label_head']
    
    total_samples = len(label_drowsy)
    print(f"Extracting spatial features for {total_samples} sequence samples (T=30)...")
    
    eye_encoder = FastSpatialEncoder(out_dim=128)
    mouth_encoder = FastSpatialEncoder(out_dim=128)
    
    if os.path.exists('checkpoints/pretrained_eye_encoder.pt'):
        print("  [Precompute] Loading checkpoints/pretrained_eye_encoder.pt...")
        eye_encoder.load_state_dict(torch.load('checkpoints/pretrained_eye_encoder.pt', map_location='cpu'))
    if os.path.exists('checkpoints/pretrained_mouth_encoder.pt'):
        print("  [Precompute] Loading checkpoints/pretrained_mouth_encoder.pt...")
        mouth_encoder.load_state_dict(torch.load('checkpoints/pretrained_mouth_encoder.pt', map_location='cpu'))
        
    eye_encoder.eval()
    mouth_encoder.eval()
    
    all_eye_feats = []
    all_mouth_feats = []
    
    # Process in batches of sequences
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, batch_size), desc=f"Encoding {os.path.basename(npz_input_path)}"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Read chunk as uint8 and normalize to float32 (0..1)
            b_eye_uint8 = torch.from_numpy(eye_imgs[start_idx:end_idx].copy()).float() / 255.0
            b_mouth_uint8 = torch.from_numpy(mouth_imgs[start_idx:end_idx].copy()).float() / 255.0
            
            B, T, C, H, W = b_eye_uint8.shape
            
            b_eye_flat = b_eye_uint8.view(B * T, C, H, W)
            b_mouth_flat = b_mouth_uint8.view(B * T, C, H, W)
            
            e_feat = eye_encoder(b_eye_flat).view(B, T, 128).cpu().numpy()
            m_feat = mouth_encoder(b_mouth_flat).view(B, T, 128).cpu().numpy()
            
            all_eye_feats.append(e_feat)
            all_mouth_feats.append(m_feat)
            
    eye_feats_arr = np.concatenate(all_eye_feats, axis=0)
    mouth_feats_arr = np.concatenate(all_mouth_feats, axis=0)
    
    print(f"Saving extracted features to {npz_output_path} (Size: ~{(eye_feats_arr.nbytes + mouth_feats_arr.nbytes)/1e6:.1f} MB)...")
    np.savez(
        npz_output_path,
        eye_feats=eye_feats_arr,
        mouth_feats=mouth_feats_arr,
        geo_vecs=np.array(geo_vecs),
        cond_idx=np.array(cond_idx),
        label_drowsy=np.array(label_drowsy),
        label_eye=np.array(label_eye),
        label_mouth=np.array(label_mouth),
        label_head=np.array(label_head)
    )
    print(f"Successfully saved {npz_output_path}!")

if __name__ == '__main__':
    extract_features_for_dataset('data_cache/train_consolidated.npz', 'data_cache/features_train.npz', batch_size=64)
    extract_features_for_dataset('data_cache/eval_consolidated.npz', 'data_cache/features_eval.npz', batch_size=64)
    extract_features_for_dataset('data_cache/test_consolidated.npz', 'data_cache/features_test.npz', batch_size=64)
