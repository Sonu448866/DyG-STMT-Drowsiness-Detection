import os
import glob
import numpy as np
from tqdm import tqdm

def consolidate_folder(shards_dir, out_file):
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
    if not shard_files:
        print(f"No shards found in {shards_dir}")
        return
        
    print(f"Consolidating {len(shard_files)} shards from {shards_dir} into {out_file}...")
    
    all_eye = []
    all_mouth = []
    all_geo = []
    all_cond = []
    all_drowsy = []
    all_l_eye = []
    all_l_mouth = []
    all_l_head = []
    
    for s_file in tqdm(shard_files, desc=f"Reading {os.path.basename(shards_dir)}"):
        try:
            data = np.load(s_file)
            all_eye.append(data['eye_imgs'])
            all_mouth.append(data['mouth_imgs'])
            all_geo.append(data['geo_vecs'])
            all_cond.append(data['cond_idx'])
            all_drowsy.append(data['label_drowsy'])
            all_l_eye.append(data['label_eye'])
            all_l_mouth.append(data['label_mouth'])
            all_l_head.append(data['label_head'])
        except Exception as e:
            continue
            
    eye_arr = np.concatenate(all_eye, axis=0)
    mouth_arr = np.concatenate(all_mouth, axis=0)
    geo_arr = np.concatenate(all_geo, axis=0)
    cond_arr = np.concatenate(all_cond, axis=0)
    drowsy_arr = np.concatenate(all_drowsy, axis=0)
    l_eye_arr = np.concatenate(all_l_eye, axis=0)
    l_mouth_arr = np.concatenate(all_l_mouth, axis=0)
    l_head_arr = np.concatenate(all_l_head, axis=0)
    
    print(f"Writing {len(drowsy_arr)} sequences to {out_file}...")
    # Use uncompressed savez for instant disk write
    np.savez(
        out_file,
        eye_imgs=eye_arr,
        mouth_imgs=mouth_arr,
        geo_vecs=geo_arr,
        cond_idx=cond_arr,
        label_drowsy=drowsy_arr,
        label_eye=l_eye_arr,
        label_mouth=l_mouth_arr,
        label_head=l_head_arr
    )
    print(f"Successfully saved {out_file} ({len(drowsy_arr)} samples).")

if __name__ == '__main__':
    consolidate_folder('data_cache/train_shards', 'data_cache/train_consolidated.npz')
    consolidate_folder('data_cache/eval_shards', 'data_cache/eval_consolidated.npz')
    consolidate_folder('data_cache/test_shards', 'data_cache/test_consolidated.npz')
