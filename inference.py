import os
import time
import argparse
from pathlib import Path
import importlib
import json
from collections import defaultdict, Counter

import torch
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from tqdm.auto import tqdm

# torchvision for transforms
from torchvision import transforms

# import architecture (we will also import into a module to assign config later)
import Deformable_SVTR_Architecture as arch  # module import, used when injecting config
from Deformable_SVTR_Architecture import SVTR  # class

# Additional dependency used in OCRMetrics
try:
    import editdistance
except Exception:
    editdistance = None
    # We'll print a helpful error later if metrics are requested.

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[inference] Using device: {device}")

# ---------- Text processor (keeps your design) ----------
class TextProcessor:
    def __init__(self, unified_charset, max_length=64):
        self.max_length = max_length
        # special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        self.BLANK_TOKEN = '<BLANK>'

        vocab = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN, self.BLANK_TOKEN] + sorted(list(unified_charset))
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode_text(self, text):
        if not isinstance(text, str): text = str(text)
        seq = [self.char2idx.get(ch, self.char2idx[self.UNK_TOKEN]) for ch in text]
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        else:
            seq += [self.char2idx[self.PAD_TOKEN]] * (self.max_length - len(seq))
        return seq

    def decode_sequence(self, sequence):
        text = ""
        for idx in sequence:
            char = self.idx2char.get(idx, self.UNK_TOKEN)
            if char not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN, self.BLANK_TOKEN]:
                text += char
        return text

# ---------- OCR Metrics class (your class, integrated) ----------
class OCRMetrics:
    """Comprehensive evaluation metrics for OCR (integrated)."""

    def __init__(self, text_processor):
        if editdistance is None:
            raise RuntimeError("editdistance package is required for OCRMetrics. Install via `pip install editdistance`.")
        self.text_processor = text_processor
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.correct_samples = 0
        self.total_chars = 0
        self.correct_chars = 0
        self.total_edit_distance = 0
        self.total_word_distance = 0
        self.total_words = 0
        self.correct_words = 0
        self.char_error_rates = []
        self.word_error_rates = []
        self.sequence_accuracies = []

    def update(self, predictions, targets):
        """Update metrics with batch predictions and targets (lists of strings)."""
        batch_size = len(predictions)
        self.total_samples += batch_size

        for pred_text, target_text in zip(predictions, targets):
            if target_text is None:
                continue  # skip if no ground truth for this sample
            # Character-level metrics
            pred_chars = list(pred_text)
            target_chars = list(target_text)

            self.total_chars += len(target_chars)

            # Character accuracy (match aligned prefix length)
            correct_chars = sum(1 for p, t in zip(pred_chars, target_chars) if p == t)
            self.correct_chars += correct_chars

            # Character Error Rate (CER)
            edit_dist = editdistance.eval(pred_text, target_text)
            self.total_edit_distance += edit_dist
            cer = edit_dist / max(len(target_text), 1)
            self.char_error_rates.append(cer)

            # Word-level metrics
            pred_words = pred_text.split()
            target_words = target_text.split()

            self.total_words += len(target_words)

            # Word accuracy
            correct_words = sum(1 for p, t in zip(pred_words, target_words) if p == t)
            self.correct_words += correct_words

            # Word Error Rate (WER)
            word_edit_dist = editdistance.eval(pred_words, target_words)
            self.total_word_distance += word_edit_dist
            wer = word_edit_dist / max(len(target_words), 1)
            self.word_error_rates.append(wer)

            # Sequence accuracy
            if pred_text == target_text:
                self.correct_samples += 1
                self.sequence_accuracies.append(1.0)
            else:
                self.sequence_accuracies.append(0.0)

    def get_metrics(self):
        """Calculate and return all metrics"""
        if self.total_samples == 0:
            return {}

        metrics = {
            'sequence_accuracy': float(self.correct_samples / max(self.total_samples, 1)),
            'character_accuracy': float(self.correct_chars / max(self.total_chars, 1)),
            'word_accuracy': float(self.correct_words / max(self.total_words, 1)),
            'character_error_rate': float(self.total_edit_distance / max(self.total_chars, 1)),
            'word_error_rate': float(self.total_word_distance / max(self.total_words, 1)),
            'avg_cer': float(np.mean(self.char_error_rates)) if len(self.char_error_rates) > 0 else 0.0,
            'avg_wer': float(np.mean(self.word_error_rates)) if len(self.word_error_rates) > 0 else 0.0,
            'avg_sequence_accuracy': float(np.mean(self.sequence_accuracies)) if len(self.sequence_accuracies) > 0 else 0.0,
            'total_samples': int(self.total_samples)
        }

        return metrics

# ---------- Utilities ----------
def ctc_decode(indices, idx2char, blank_token):
    """Greedy collapse + remove blanks (keeps same semantics as earlier)."""
    decoded = []
    prev = None
    for idx in indices:
        ch = idx2char.get(int(idx), blank_token)
        if ch != blank_token and ch != prev:
            decoded.append(ch)
        prev = ch
    return ''.join(decoded)

def load_model(model_path):
    """
    Loads checkpoint, instantiates model and injects config into architecture module namespace
    so the SVTR.forward() which references config[...] still works.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"[load_model] Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    cfg = checkpoint['config']
    text_processor = checkpoint.get('text_processor', None)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint.get('state_dict', checkpoint)

    model = SVTR(
        img_size=(cfg['img_height'], cfg['img_width']),
        in_chans=cfg['channels'],
        vocab_size=cfg['vocab_size'],
        local_type=cfg['local_type'],
        embed_dims=cfg['embed_dims'],
        heads=cfg['heads'],
        mlp_ratio=cfg['mlp_ratio'],
        window_sizes=cfg['window_sizes'],
        num_blocks=cfg['num_blocks'],
        pattern=cfg['pattern'],
        drop=cfg.get('dropout_rate', 0.0),
        n_points=cfg.get('n_points', 9),
        offset_scale=cfg.get('offset_scale', 4.0),
    ).to(device).eval()

    model.load_state_dict(state)
    # Inject config into the architecture module so SVTR.forward() can find it as a global
    try:
        setattr(arch, 'config', cfg)
    except Exception:
        globals()['config'] = cfg

    print("[load_model] model ready.")
    return model, cfg, text_processor

# ---------- Preprocessing (batch friendly) ----------
def build_transform(img_height, img_width, mean, std, keep_aspect=True):
    # Keep aspect ratio + pad (pad color chosen by mean scaled to 0..255)
    def preprocess_pil(img: Image.Image):
        img = img.convert('RGB')
        if keep_aspect:
            w0, h0 = img.size
            scale = min(img_width / w0, img_height / h0)
            new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # pad right/bottom (left/top anchored)
            background = Image.new('RGB', (img_width, img_height),
                                   (int(mean[0] * 255), int(mean[1] * 255), int(mean[2] * 255)))
            background.paste(img, (0, 0))
            img = background
        else:
            img = img.resize((img_width, img_height), Image.BILINEAR)

        return img

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return preprocess_pil, to_tensor

def preprocess_batch(image_paths, config, keep_aspect=True):
    preprocess_pil, to_tensor = build_transform(config['img_height'], config['img_width'],
                                                config['dataset_mean'], config['dataset_std'], keep_aspect=keep_aspect)
    tensors = []
    for p in image_paths:
        pil = preprocess_pil(Image.open(p))
        t = to_tensor(pil)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).to(device)
    return batch

# ---------- CTC beam search (prefix beam search, log-domain) ----------
def _logsumexp(a, b):
    return np.logaddexp(a, b)

def _log_softmax_numpy(logits):
    # logits: (T, V)
    # numerically stable log-softmax per row
    maxes = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - maxes)
    sums = np.sum(exps, axis=1, keepdims=True)
    return (logits - maxes) - np.log(sums)  # log softmax

def ctc_beam_search_single(logits_np, beam_size, blank_idx, idx2char, ignore_tokens=set()):
    """
    logits_np: numpy array (T, V) raw logits (not log-softmax)
    returns: decoded string (best prefix)
    NOTE: We prune candidate symbols per step to topk to keep speed reasonable.
    """
    T, V = logits_np.shape
    logp = _log_softmax_numpy(logits_np)  # (T, V)

    neg_inf = -1e9
    # beam: mapping prefix(tuple of ints) -> (log_pb, log_pnb)
    beam = {(): (0.0, neg_inf)}  # empty prefix: prob=1 (log 0)

    # Precompute topk tokens per time-step to limit expansions (2*beam_size is common)
    topk_tokens_per_t = []
    k_candidates = max(10, beam_size * 2)  # safe default
    for t in range(T):
        # pick top-k tokens by logp[t]
        order = np.argsort(-logp[t])[:min(V, k_candidates)]
        topk_tokens_per_t.append(order)

    for t in range(T):
        new_beam = {}
        topk = topk_tokens_per_t[t]
        logp_t = logp[t]

        for prefix, (log_pb, log_pnb) in beam.items():
            # total prob for prefix so far
            log_total = np.logaddexp(log_pb, log_pnb)

            # 1) extend by blank -> prefix stays the same
            l_pb_old, l_pnb_old = new_beam.get(prefix, (neg_inf, neg_inf))
            candidate = log_total + float(logp_t[blank_idx])
            l_pb_new = np.logaddexp(l_pb_old, candidate)
            new_beam[prefix] = (l_pb_new, l_pnb_old)

            # 2) extend by non-blank tokens (only topk)
            for c in topk:
                if c == blank_idx:
                    continue
                logpc = float(logp_t[c])
                new_prefix = prefix + (int(c),)
                # Case: last char of prefix is same as c
                if len(prefix) > 0 and prefix[-1] == c:
                    # only previous blank-ending paths can be extended with same char
                    val = log_pb + logpc
                else:
                    val = log_total + logpc

                l_pb_old, l_pnb_old = new_beam.get(new_prefix, (neg_inf, neg_inf))
                l_pnb_new = np.logaddexp(l_pnb_old, val)
                new_beam[new_prefix] = (l_pb_old, l_pnb_new)

        # prune to top beam_size prefixes
        scored = []
        for pref, (lpb, lpnb) in new_beam.items():
            total = np.logaddexp(lpb, lpnb)
            scored.append((total, pref, (lpb, lpnb)))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:beam_size]
        beam = {pref: scores for (_, pref, scores) in scored}

    # pick best prefix
    best_prefix = max(beam.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]  # tuple of ints

    # convert prefix indices -> chars and postprocess
    chars = []
    prev = None
    for idx in best_prefix:
        ch = idx2char.get(int(idx), None)
        if ch is None:
            continue
        if ch in ignore_tokens or ch == '':
            continue
        # collapse duplicates (just in case)
        if ch == prev:
            # skip (CTC would have required a blank to generate repeat)
            continue
        chars.append(ch)
        prev = ch

    return ''.join(chars)

def ctc_beam_search_batch(preds, text_processor, beam_size=4):
    """
    preds: torch.Tensor (B, W, V) logits
    returns: list[str] length B (decoded)
    """
    results = []
    blank_idx = text_processor.char2idx[text_processor.BLANK_TOKEN]
    idx2char = text_processor.idx2char
    # tokens to ignore in final mapping
    ignore_tokens = {text_processor.PAD_TOKEN, text_processor.START_TOKEN, text_processor.END_TOKEN,
                     text_processor.UNK_TOKEN, text_processor.BLANK_TOKEN}

    with torch.no_grad():
        preds_cpu = preds.cpu().numpy()  # (B, W, V)
    for i in range(preds_cpu.shape[0]):
        logits_np = preds_cpu[i]  # (W, V)
        decoded = ctc_beam_search_single(logits_np, beam_size, blank_idx, idx2char, ignore_tokens=ignore_tokens)
        results.append(decoded)
    return results

# ---------- Decoding ----------
def decode_predictions_batch(preds, text_processor, method='greedy', beam_size=4):
    """
    preds: (B, W, vocab_size) logits
    method: 'greedy' (default) or 'beam'
    """
    if method == 'greedy':
        # greedy argmax (existing behavior)
        with torch.no_grad():
            max_inds = preds.argmax(dim=2)        # (B, W)
        results = []
        for i in range(max_inds.shape[0]):
            inds = max_inds[i].cpu().numpy().tolist()
            txt = ctc_decode(inds, text_processor.idx2char, blank_token=text_processor.BLANK_TOKEN)
            results.append(txt)
        return results
    elif method == 'beam':
        return ctc_beam_search_batch(preds, text_processor, beam_size=beam_size)
    else:
        raise ValueError("Unknown decode method: choose 'greedy' or 'beam'")

# ---------- Visualization ----------
def visualize_and_save(image_path, text, out_path=None):
    img = Image.open(image_path).convert('RGB')
    display_text = get_display(arabic_reshaper.reshape(text))
    fig = plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.title(display_text, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_and_save_metrics(ocrm: OCRMetrics, output_dir: str):
    """Create several plots for CER, WER, prediction lengths and sequence accuracy distribution."""
    metrics = ocrm.get_metrics()
    # Save metrics json
    with open(os.path.join(output_dir, 'ocr_metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Plot CER histogram
    if len(ocrm.char_error_rates) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(ocrm.char_error_rates, bins=30)
        plt.title('CER histogram')
        plt.xlabel('CER')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cer_hist.png'))
        plt.close()

    # Plot WER histogram
    if len(ocrm.word_error_rates) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(ocrm.word_error_rates, bins=30)
        plt.title('WER histogram')
        plt.xlabel('WER')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'wer_hist.png'))
        plt.close()

    # Sequence accuracy pie / bar
    if len(ocrm.sequence_accuracies) > 0:
        acc_count = sum(ocrm.sequence_accuracies)
        tot = len(ocrm.sequence_accuracies)
        plt.figure(figsize=(4,4))
        plt.pie([acc_count, tot - acc_count], labels=['Exact match','Mismatch'], autopct='%1.1f%%')
        plt.title('Sequence accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_accuracy_pie.png'))
        plt.close()

    # Prediction length histogram (derived from char_error_rates length of predictions not directly stored)
    # Create a synthetic distribution using CER lists length and total_chars if present
    # Instead, we'll save a scatter of CER vs index
    if len(ocrm.char_error_rates) > 0:
        plt.figure(figsize=(8,4))
        plt.plot(ocrm.char_error_rates, marker='.', linestyle='none', markersize=4)
        plt.title('CER per sample (index order)')
        plt.xlabel('sample idx')
        plt.ylabel('CER')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cer_per_sample.png'))
        plt.close()

    print(f"[plot] Saved metric plots to {output_dir}")

# ---------- Labels file reader ----------
def read_labels_file(labels_file):
    """
    Support lines of form:
      <image_path_or_name>\\t<ground_truth_text>
    returns: dict mapping filename (basename) and fullpath -> ground_truth
    """
    mapping = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support tab or first whitespace separation
            if '\t' in line:
                key, val = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                else:
                    # if only one column, we treat this as ground-truth and rely on order later
                    key, val = None, parts[0]
            if key is None:
                # store under special numeric key behavior is handled in main if necessary
                continue
            mapping[key] = val
            mapping[os.path.basename(key)] = val
            # also keep normalized path without leading ./ or /
            mapping[str(Path(key).resolve())] = val
    return mapping

# ---------- Inference helpers ----------
def inference_single(model, image_path, text_processor, config, keep_aspect=True, method='greedy', beam_size=4):
    x = preprocess_batch([image_path], config, keep_aspect=keep_aspect)
    with torch.no_grad():
        preds = model(x)    # (1, W, vocab_size)
    decoded = decode_predictions_batch(preds, text_processor, method=method, beam_size=beam_size)[0]
    return decoded

def inference_batch(model, image_paths, text_processor, config, keep_aspect=True, batch_size=8, method='greedy', beam_size=4):
    results = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i+batch_size]
        x = preprocess_batch(chunk, config, keep_aspect=keep_aspect)
        with torch.no_grad():
            preds = model(x)    # (B, W, vocab_size)
        decoded = decode_predictions_batch(preds, text_processor, method=method, beam_size=beam_size)
        results.extend(decoded)
    return results

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Optimized Deformable SVTR Inference (with analysis & beam-search)")
    parser.add_argument('--model_path', required=True, help='checkpoint path (.pth/.pt)')
    parser.add_argument('--image_path', required=True, help='single image or a folder containing images')
    parser.add_argument('--output_dir', default='./results', help='where to save results')
    parser.add_argument('--visualize', action='store_true', help='save visualization image with predicted text')
    parser.add_argument('--keep_aspect', action='store_true', help='preserve aspect ratio and pad (recommended)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for batch inference')
    parser.add_argument('--analyze', action='store_true', help='Compute OCR metrics and plots (requires --labels_file)')
    parser.add_argument('--labels_file', type=str, default=None, help='Optional labels file (image_path\\tground_truth per line)')
    parser.add_argument('--use_beam', action='store_true', help='Use CTC beam search decoding instead of greedy')
    parser.add_argument('--beam_size', type=int, default=4, help='Beam size for CTC beam search')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, config, tp = load_model(args.model_path)
    if tp is None:
        # If the checkpoint did not include a text_processor object, construct it from permissible_chars
        if 'permissible_chars' in config:
            tp = TextProcessor(config['permissible_chars'], max_length=config.get('max_text_length', 64))
        else:
            raise RuntimeError("No text_processor found in checkpoint and config lacks permissible_chars")

    # build image list
    image_path = Path(args.image_path)
    if image_path.is_dir():
        image_paths = sorted([str(p) for p in image_path.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    else:
        image_paths = [str(image_path)]

    print(f"[main] Running inference on {len(image_paths)} image(s). Batch size: {args.batch_size}")

    method = 'beam' if args.use_beam else 'greedy'
    results = inference_batch(model, image_paths, tp, config, keep_aspect=args.keep_aspect,
                              batch_size=args.batch_size, method=method, beam_size=args.beam_size)

    # Save results
    out_txt = os.path.join(args.output_dir, "inference_results.txt")
    with open(out_txt, 'w', encoding='utf-8') as f:
        for p, txt in zip(image_paths, results):
            f.write(f"{p}\t{txt}\n")
    print(f"[main] Results saved to {out_txt}")

    # optional visualizations
    if args.visualize:
        for p, txt in zip(image_paths, results):
            name = Path(p).stem
            out_img = os.path.join(args.output_dir, f"{name}_pred.png")
            visualize_and_save(p, txt, out_img)
        print(f"[main] Visualizations saved to {args.output_dir}")

    # pretty print first few results
    for p, txt in zip(image_paths[:10], results[:10]):
        print(f"{Path(p).name} -> {get_display(arabic_reshaper.reshape(txt))}")

    # ---------- Analysis if requested ----------
    if args.analyze:
        # Ensure editdistance is available
        if editdistance is None:
            raise RuntimeError("To run --analyze you must install `editdistance` (pip install editdistance).")

        if args.labels_file is None:
            # try to find labels file in the same folder as image_path
            candidate = os.path.join(str(image_path), 'labels.txt') if image_path.is_dir() else None
            if candidate and os.path.exists(candidate):
                args.labels_file = candidate
                print(f"[analyze] Found labels file at {candidate}")
            else:
                raise RuntimeError("You requested --analyze but didn't supply --labels_file and none found automatically.")

        mapping = read_labels_file(args.labels_file)
        # Build targets list aligned with image_paths
        targets = []
        missing = 0
        for p in image_paths:
            # try exact path, basename, resolved path
            val = mapping.get(p, None)
            if val is None:
                val = mapping.get(os.path.basename(p), None)
            if val is None:
                val = mapping.get(str(Path(p).resolve()), None)
            if val is None:
                missing += 1
            targets.append(val)

        print(f"[analyze] Found ground truth for {len(image_paths)-missing}/{len(image_paths)} samples.")
        # compute metrics only for samples where target exists
        valid_preds = []
        valid_targets = []
        for pred, tgt in zip(results, targets):
            if tgt is not None:
                valid_preds.append(pred)
                valid_targets.append(tgt)

        ocrm = OCRMetrics(tp)
        ocrm.update(valid_preds, valid_targets)
        metrics = ocrm.get_metrics()
        print("[analyze] OCR metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Save metrics and plots
        with open(os.path.join(args.output_dir, 'ocr_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        plot_and_save_metrics(ocrm, args.output_dir)

if __name__ == "__main__":
    main()


# python inference.py   --model_path models/Defo_SVTR_Large_2.3.pth   --image_path test_dataset_ocr/ar/   --output_dir results/ar --visualize --analyze --labels_file test_dataset_ocr/ar/ar_ground_truth.txt   --use_beam --beam_size 20 --batch_size 4 --keep_aspect

# python inference.py \
#   --model_path models/Defo_SVTR_Large_1.0.pth \
#   --image_path test_dataset_ocr/ar/ \
#   --output_dir results/ar \
#   --visualize \
#   --keep_aspect \
#   --batch_size 4

# python inference.py \
#   --model_path models/Defo_SVTR_Large_2.3.pth \
#   --image_path test_dataset_ocr/ar/ \
#   --output_dir results/ar \
#   --use_beam --beam_size 4 \
#   --keep_aspect --batch_size 2

# python inference.py \
#   --model_path models/Defo_SVTR_Large_2.3.pth \
#   --image_path test_dataset_ocr/ar/ \
#   --output_dir results/ar \
#   --analyze --labels_file test_dataset_ocr/ar/ar_ground_truth.txt \
#   --use_beam --beam_size 4 --batch_size 4 --keep_aspect
