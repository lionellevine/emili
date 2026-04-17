# personalized facial emotion recognition pipeline for EMILI
# steps: collect user data -> baseline evaluation -> fine tune -> evaluate

import argparse
import csv
from datetime import datetime
import json
import os
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SiglipForImageClassification


MODEL_ID = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
TARGET_EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprised']
INTENSITY_EMOTIONS = ['angry', 'happy', 'sad', 'surprised']
TARGET_TO_SIGLIP = {'angry': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprised': 5}
SIGLIP_INDICES = [TARGET_TO_SIGLIP[e] for e in TARGET_EMOTIONS]


def now_ms():
    return int(time.time() * 1000)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_prompts(prompt_path):
    with open(prompt_path, 'r') as file:
        prompts = json.load(file)
    return prompts


def project_paths(base_dir, user_id, session_id):
    user_root = os.path.join(base_dir, user_id)
    frames_root = os.path.join(user_root, 'frames', session_id)
    models_root = os.path.join(user_root, 'models')
    reports_root = os.path.join(user_root, 'reports')
    metadata_root = os.path.join(user_root, 'metadata')

    for path in [user_root, frames_root, models_root, reports_root, metadata_root]:
        ensure_dir(path)

    paths = {
        'user_root': user_root,
        'frames_root': frames_root,
        'models_root': models_root,
        'reports_root': reports_root,
        'metadata_root': metadata_root,
        'dataset_csv': os.path.join(metadata_root, 'dataset.csv'),
        'split_json': os.path.join(metadata_root, 'split.json'),
        'baseline_csv': os.path.join(reports_root, f'baseline_{session_id}.csv'),
        'baseline_metrics_json': os.path.join(reports_root, f'baseline_metrics_{session_id}.json'),
        'personalized_csv': os.path.join(reports_root, f'personalized_{session_id}.csv'),
        'personalized_metrics_json': os.path.join(reports_root, f'personalized_metrics_{session_id}.json'),
        'checkpoint_path': os.path.join(models_root, 'personalized_siglip2.pt'),
        'intensity_path': os.path.join(models_root, 'intensity_calibrator.json')
    }
    return paths


def open_camera(camera_id):
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_id}")
    return camera


def draw_overlay(frame, lines, footer=None):
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    y = 30
    for line in lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 255, 30), 2, cv2.LINE_AA)
        y += 30

    if footer:
        cv2.putText(overlay, footer, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def show_collection_intro(camera, prompts, seconds_per_prompt, user_id, frames_root):
    unique_emotions = sorted(list(set([p['emotion'] for p in prompts])))
    non_neutral_levels = sorted(list(set([int(p['intensity']) for p in prompts if p['emotion'] != 'neutral'])))
    neutral_levels = sorted(list(set([int(p['intensity']) for p in prompts if p['emotion'] == 'neutral'])))
    num_prompts = len(prompts)
    estimated_seconds = num_prompts * seconds_per_prompt
    estimated_minutes = estimated_seconds / 60.0

    if len(neutral_levels) <= 1:
        neutral_note = "neutral has one level only (no intensity scale shown)"
    else:
        neutral_note = "neutral has multiple levels in this prompt file"

    while True:
        success, frame_bgr = camera.read()
        if not success:
            continue

        lines = [
            "Personalized FER data collection",
            f"User: {user_id}",
            f"Emotions recorded: {', '.join(unique_emotions)}",
            f"Total prompts: {num_prompts}",
            f"Recording length per prompt: {seconds_per_prompt} seconds",
            f"Estimated recording time: ~{estimated_minutes:0.1f} minutes",
            f"Intensity levels for non-neutral: {non_neutral_levels}",
            f"Neutral rule: {neutral_note}",
            "During recording the app auto-labels each captured frame.",
            f"Privacy: frames are stored locally at {frames_root}",
            "Controls: SPACE=start prompt, S=skip prompt, Q=quit collection",
            "Press C to continue to the first prompt, or Q to exit now."
        ]
        shown = draw_overlay(frame_bgr, lines)
        cv2.imshow('EMILI Personalization Capture', shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('c'):
            return True


def largest_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    return faces[idx]


def read_existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    return rows


def write_rows(csv_path, rows):
    fieldnames = [
        'user_id', 'session_id', 'frame_path', 'timestamp_ms', 'emotion_label', 'intensity_label',
        'prompt_id', 'prompt_text', 'split', 'face_detected'
    ]
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_rows(csv_path, new_rows):
    rows = read_existing_rows(csv_path)
    rows.extend(new_rows)
    write_rows(csv_path, rows)


def safe_user_text(text):
    return ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in text)


def collect_data(user_id, camera_id, prompts, paths, seconds_per_prompt=10, sample_fps=5):
    print("Starting guided data collection.")
    print("Window controls: SPACE=start prompt, S=skip prompt, Q=quit collection")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = open_camera(camera_id)

    new_rows = []
    frame_interval = 1.0 / max(1, sample_fps)

    try:
        if not show_collection_intro(camera, prompts, seconds_per_prompt, user_id, paths['frames_root']):
            return new_rows

        for i, prompt in enumerate(prompts):
            emotion = prompt['emotion']
            intensity = int(prompt['intensity'])
            prompt_text = prompt['prompt']
            prompt_id = f"{emotion}_{intensity}_{i:03d}"
            intensity_text = f"{intensity}/5"
            if emotion == 'neutral':
                intensity_text = "N/A"

            waiting = True
            while waiting:
                success, frame_bgr = camera.read()
                if not success:
                    continue
                lines = [
                    f"Prompt {i + 1}/{len(prompts)}",
                    f"Emotion: {emotion}  Intensity: {intensity_text}",
                    prompt_text
                ]
                shown = draw_overlay(frame_bgr, lines, footer='SPACE=start | S=skip | Q=quit')
                cv2.imshow('EMILI Personalization Capture', shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return new_rows
                if key == ord('s'):
                    waiting = False
                    prompt_id = None
                if key == ord(' '):
                    waiting = False

            if prompt_id is None:
                continue

            start = time.time()
            next_capture = start
            while (time.time() - start) < seconds_per_prompt:
                success, frame_bgr = camera.read()
                if not success:
                    continue

                elapsed = time.time() - start
                remaining = max(0.0, seconds_per_prompt - elapsed)
                lines = [
                    f"Recording: {emotion} intensity {intensity_text}",
                    prompt_text,
                    f"Time left: {remaining:0.1f}s"
                ]
                shown = draw_overlay(frame_bgr, lines, footer='Q=quit this session')
                cv2.imshow('EMILI Personalization Capture', shown)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return new_rows

                if time.time() >= next_capture:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    face = largest_face(gray, face_cascade)

                    face_detected = '0'
                    crop = frame_rgb
                    if face is not None:
                        x, y, w, h = face
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(frame_rgb.shape[1], x + w)
                        y2 = min(frame_rgb.shape[0], y + h)
                        crop = frame_rgb[y1:y2, x1:x2]
                        face_detected = '1'

                    ts = now_ms()
                    filename = f"{safe_user_text(user_id)}_{prompt_id}_{ts}.jpg"
                    frame_path = os.path.join(paths['frames_root'], filename)
                    Image.fromarray(crop).save(frame_path, quality=95)

                    new_rows.append({
                        'user_id': user_id,
                        'session_id': os.path.basename(paths['frames_root']),
                        'frame_path': frame_path,
                        'timestamp_ms': str(ts),
                        'emotion_label': emotion,
                        'intensity_label': str(intensity),
                        'prompt_id': prompt_id,
                        'prompt_text': prompt_text,
                        'split': '',
                        'face_detected': face_detected
                    })
                    next_capture += frame_interval

    finally:
        camera.release()
        cv2.destroyAllWindows()

    return new_rows


class PersonalizationDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row['frame_path']).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)
        emotion_idx = TARGET_EMOTIONS.index(row['emotion_label'])
        intensity = float(row['intensity_label'])
        return pixel_values, emotion_idx, intensity


def collate_batch(batch):
    pixels = torch.stack([item[0] for item in batch], dim=0)
    emotions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    intensities = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return pixels, emotions, intensities


def split_rows(rows, split_json_path, train_ratio=0.8):
    buckets = {}
    for idx, row in enumerate(rows):
        key = f"{row['emotion_label']}|{row['intensity_label']}"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(idx)

    random.seed(1337)
    train_idx, val_idx = [], []
    for key in buckets:
        idxs = buckets[key]
        random.shuffle(idxs)
        cut = int(round(len(idxs) * train_ratio))
        cut = min(max(cut, 1), len(idxs) - 1) if len(idxs) > 1 else 1
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])

    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = [train_idx.pop()]

    for idx in train_idx:
        rows[idx]['split'] = 'train'
    for idx in val_idx:
        rows[idx]['split'] = 'val'

    split_info = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'train_count': len(train_idx),
        'val_count': len(val_idx),
        'train_ratio': train_ratio
    }
    with open(split_json_path, 'w') as file:
        json.dump(split_info, file, indent=2)

    return rows, split_info


def load_model(device, checkpoint_path=None):
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in payload:
            model.load_state_dict(payload['model_state_dict'])
            print(f"Loaded personalized checkpoint from {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return processor, model


def target_probs_from_logits(logits):
    subset = logits[..., SIGLIP_INDICES]
    probs = torch.softmax(subset, dim=-1)
    return probs


def evaluate_rows(rows, processor, model, device, output_csv_path, metrics_json_path, intensity_calibrator=None):
    header = [
        'frame_path', 'true_label', 'true_intensity', 'predicted_label', 'predicted_intensity', 'correct_label',
        'prob_angry', 'prob_happy', 'prob_neutral', 'prob_sad', 'prob_surprised'
    ]
    records = []

    correct = 0
    true_labels = []
    pred_labels = []
    intensity_errors = []

    for row in rows:
        image = Image.open(row['frame_path']).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = target_probs_from_logits(logits).squeeze(0).detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = TARGET_EMOTIONS[pred_idx]
        true_label = row['emotion_label']
        true_intensity = float(row['intensity_label'])
        pred_prob = float(probs[pred_idx])
        sorted_probs = np.sort(probs)
        top2 = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        intensity_signal = 0.7 * pred_prob + 0.3 * max(0.0, pred_prob - top2)

        if pred_label == 'neutral':
            pred_intensity = None
        elif intensity_calibrator and pred_label in intensity_calibrator and intensity_calibrator[pred_label].get('mode') == 'bounded_linear':
            coeff = intensity_calibrator[pred_label]
            p10 = coeff['p10']
            p90 = coeff['p90']
            denom = max(1e-6, p90 - p10)
            z = np.clip((intensity_signal - p10) / denom, 0.0, 1.0)
            pred_intensity = coeff['a'] * z + coeff['b']
        else:
            pred_intensity = 1.0 + 4.0 * intensity_signal
        if pred_intensity is not None:
            pred_intensity = float(np.clip(pred_intensity, 1.0, 5.0))

        is_correct = int(pred_label == true_label)
        correct += is_correct

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        if true_label in INTENSITY_EMOTIONS and pred_intensity is not None:
            intensity_errors.append(abs(pred_intensity - true_intensity))

        records.append([
            row['frame_path'],
            true_label,
            "N/A" if true_label == 'neutral' else int(true_intensity),
            pred_label,
            "N/A" if pred_intensity is None else f"{pred_intensity:0.3f}",
            is_correct,
            f"{probs[0]:0.6f}",
            f"{probs[1]:0.6f}",
            f"{probs[2]:0.6f}",
            f"{probs[3]:0.6f}",
            f"{probs[4]:0.6f}"
        ])

    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(records)

    macro_f1, per_class = macro_f1_score(true_labels, pred_labels, TARGET_EMOTIONS)
    metrics = {
        'num_samples': len(rows),
        'accuracy': float(correct / max(1, len(rows))),
        'macro_f1': float(macro_f1),
        'intensity_mae': float(np.mean(intensity_errors) if len(intensity_errors) > 0 else 0.0),
        'per_class_f1': per_class
    }
    with open(metrics_json_path, 'w') as file:
        json.dump(metrics, file, indent=2)

    return metrics


def macro_f1_score(y_true, y_pred, labels):
    per_class = {}
    f1s = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if p == label and t == label:
                tp += 1
            elif p == label and t != label:
                fp += 1
            elif p != label and t == label:
                fn += 1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = float(f1)
        f1s.append(f1)
    return float(np.mean(f1s) if len(f1s) > 0 else 0.0), per_class


def train_personalized_model(rows, checkpoint_path, device, epochs=4, batch_size=8, lr=2e-5):
    train_rows = [row for row in rows if row['split'] == 'train']
    val_rows = [row for row in rows if row['split'] == 'val']

    if len(train_rows) < 5:
        raise RuntimeError("Need at least 5 training samples to fine-tune.")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID).to(device)

    train_dataset = PersonalizationDataset(train_rows, processor)
    val_dataset = PersonalizationDataset(val_rows, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for pixels, emotions, intensities in train_loader:
            del intensities
            pixels = pixels.to(device)
            emotions = emotions.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=pixels).logits
            target_logits = logits[:, SIGLIP_INDICES]
            loss = ce_loss(target_logits, emotions)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        val_true = []
        val_pred = []
        model.eval()
        with torch.no_grad():
            for pixels, emotions, intensities in val_loader:
                del intensities
                pixels = pixels.to(device)
                logits = model(pixel_values=pixels).logits
                probs = torch.softmax(logits[:, SIGLIP_INDICES], dim=-1)
                preds = torch.argmax(probs, dim=-1).detach().cpu().numpy().tolist()
                trues = emotions.detach().cpu().numpy().tolist()
                val_true.extend([TARGET_EMOTIONS[t] for t in trues])
                val_pred.extend([TARGET_EMOTIONS[p] for p in preds])

        macro_f1, _ = macro_f1_score(val_true, val_pred, TARGET_EMOTIONS)
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={avg_loss:0.4f} | val_macro_f1={macro_f1:0.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    payload = {
        'model_state_dict': best_state,
        'target_emotions': TARGET_EMOTIONS,
        'siglip_indices': SIGLIP_INDICES,
        'model_id': MODEL_ID,
        'saved_at': datetime.now().isoformat()
    }
    torch.save(payload, checkpoint_path)
    print(f"Saved personalized model to {checkpoint_path}")


def fit_intensity_calibrator(rows, processor, model, device, intensity_path):
    train_rows = [row for row in rows if row['split'] == 'train']
    per_emotion_x = {emotion: [] for emotion in INTENSITY_EMOTIONS}
    per_emotion_y = {emotion: [] for emotion in INTENSITY_EMOTIONS}

    model.eval()
    for row in train_rows:
        image = Image.open(row['frame_path']).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = target_probs_from_logits(logits).squeeze(0).detach().cpu().numpy()

        emotion = row['emotion_label']
        if emotion not in INTENSITY_EMOTIONS:
            continue
        pred_idx = int(np.argmax(probs))
        pred_label = TARGET_EMOTIONS[pred_idx]
        if pred_label != emotion:
            continue
        emo_idx = TARGET_EMOTIONS.index(emotion)
        sorted_probs = np.sort(probs)
        top1 = float(probs[emo_idx])
        top2 = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        signal = 0.7 * top1 + 0.3 * max(0.0, top1 - top2)
        per_emotion_x[emotion].append(signal)
        per_emotion_y[emotion].append(float(row['intensity_label']))

    calibrator = {}
    for emotion in INTENSITY_EMOTIONS:
        xs = np.array(per_emotion_x[emotion], dtype=np.float64)
        ys = np.array(per_emotion_y[emotion], dtype=np.float64)
        if len(xs) >= 4 and np.std(xs) > 1e-8:
            p10 = float(np.quantile(xs, 0.10))
            p90 = float(np.quantile(xs, 0.90))
            denom = max(1e-6, p90 - p10)
            z = np.clip((xs - p10) / denom, 0.0, 1.0)
            A = np.vstack([z, np.ones_like(z)]).T
            a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
            a = float(np.clip(a, 0.0, 4.0))
            b = float(np.clip(b, 1.0, 5.0))
            calibrator[emotion] = {'mode': 'bounded_linear', 'p10': p10, 'p90': p90, 'a': a, 'b': b}
        elif len(xs) >= 1:
            calibrator[emotion] = {'mode': 'bounded_linear', 'p10': float(xs[0]), 'p90': float(xs[0] + 1e-6), 'a': 4.0, 'b': 1.0}
        else:
            calibrator[emotion] = {'mode': 'bounded_linear', 'p10': 0.0, 'p90': 1.0, 'a': 4.0, 'b': 1.0}

    calibrator['neutral'] = {'mode': 'none'}

    with open(intensity_path, 'w') as file:
        json.dump(calibrator, file, indent=2)
    print(f"Saved intensity calibrator to {intensity_path}")
    return calibrator


def main():
    parser = argparse.ArgumentParser(description='Personalized FER pipeline for EMILI (SigLIP2)')
    parser.add_argument('--user_id', type=str, required=True, help='User identifier for personalized model/data')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'collect', 'baseline', 'finetune', 'evaluate'],
                        help='Pipeline stage to run')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device id')
    parser.add_argument('--seconds_per_prompt', type=int, default=6, help='Seconds to record per prompt')
    parser.add_argument('--sample_fps', type=int, default=5, help='Captured frames per second during collection')
    parser.add_argument('--epochs', type=int, default=4, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Fine-tuning batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Fine-tuning learning rate')
    parser.add_argument('--data_root', type=str, default='data/personalization', help='Base personalization directory')
    parser.add_argument('--prompt_file', type=str, default='prompts_personalization.json', help='Prompt definition JSON')
    args = parser.parse_args()

    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = project_paths(args.data_root, args.user_id, session_id)

    rows = read_existing_rows(paths['dataset_csv'])

    if args.mode in ['all', 'collect']:
        prompts = load_prompts(args.prompt_file)
        new_rows = collect_data(
            args.user_id,
            args.camera_id,
            prompts,
            paths,
            seconds_per_prompt=args.seconds_per_prompt,
            sample_fps=args.sample_fps
        )
        append_rows(paths['dataset_csv'], new_rows)
        rows = read_existing_rows(paths['dataset_csv'])
        print(f"Collected {len(new_rows)} new samples. Total samples: {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("No personalization data found. Run collect mode first.")

    usable_rows = [row for row in rows if row.get('face_detected', '1') == '1']
    if len(usable_rows) == 0:
        raise RuntimeError("No usable samples with detected face found. Re-run collection with better lighting/framing.")
    print(f"Usable samples with detected face: {len(usable_rows)} / {len(rows)}")

    if args.mode in ['all', 'finetune']:
        usable_rows, split_info = split_rows(usable_rows, paths['split_json'], train_ratio=0.8)
        indexed = {row['frame_path']: row for row in usable_rows}
        for i, row in enumerate(rows):
            if row['frame_path'] in indexed:
                rows[i]['split'] = indexed[row['frame_path']]['split']
        write_rows(paths['dataset_csv'], rows)
        print(f"Split complete: train={split_info['train_count']} val={split_info['val_count']}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if args.mode in ['all', 'baseline']:
        processor, model = load_model(device)
        baseline_metrics = evaluate_rows(
            usable_rows,
            processor,
            model,
            device,
            paths['baseline_csv'],
            paths['baseline_metrics_json'],
            intensity_calibrator=None
        )
        print("Baseline metrics:", baseline_metrics)

    if args.mode in ['all', 'finetune']:
        train_personalized_model(
            usable_rows,
            paths['checkpoint_path'],
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate
        )

    if args.mode in ['all', 'evaluate', 'finetune']:
        processor, model = load_model(device, paths['checkpoint_path'])
        if args.mode == 'evaluate' and os.path.exists(paths['intensity_path']):
            with open(paths['intensity_path'], 'r') as file:
                calibrator = json.load(file)
            print(f"Loaded intensity calibrator from {paths['intensity_path']}")
        else:
            calibrator = fit_intensity_calibrator(usable_rows, processor, model, device, paths['intensity_path'])
        personalized_metrics = evaluate_rows(
            usable_rows,
            processor,
            model,
            device,
            paths['personalized_csv'],
            paths['personalized_metrics_json'],
            intensity_calibrator=calibrator
        )
        print("Personalized metrics:", personalized_metrics)


if __name__ == '__main__':
    main()
