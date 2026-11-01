"""
Run this file to train and evaluate all three orchid classifiers locally.

The script creates a new timestamped folder under experiments/, keeps the original
dataset unchanged, and opens an HTML report after training finishes.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import traceback
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        from train_experiment import experiment
        import torch
    except ModuleNotFoundError as exc:
        print(f'ขาดไลบรารี: {exc.name}')
        print('กรุณารัน: python -m pip install -r training/requirements.txt')
        input('กด Enter เพื่อปิด...')
        raise SystemExit(1)

    has_cuda = torch.cuda.is_available()
    run_name = datetime.now().strftime('local_%Y%m%d_%H%M%S')
    output = ROOT / 'experiments' / run_name
    args = SimpleNamespace(
        data=ROOT / 'data' / 'dataset',
        output=output,
        models=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224'],
        epochs=10,
        batch_size=32 if has_cuda else 16,
        lr=None,
        patience=5,
        val_fraction=0.20,
        seed=42,
        workers=0,       # Safe default for Windows.
        threads=4,
        device='auto',   # CUDA when available, otherwise CPU.
        latency_repeats=30,
        # Full fine-tuning on CUDA. CPU mode trains all classification heads so
        # the three-model experiment can finish on an ordinary computer.
        head_only=not has_cuda,
        prepare_only=False,
    )
    print('=' * 72)
    print('THAI ORCHID CLASSIFICATION - LOCAL TRAINING')
    print('Models: ResNet-50, EfficientNet-B0, ViT-B/16')
    print(f'Device: {"CUDA" if has_cuda else "CPU"}')
    print(f'Training mode: {"full fine-tuning" if has_cuda else "classification heads (CPU mode)"}')
    print(f'Output: {output}')
    print('กด Ctrl+C เมื่อต้องการหยุด การรันบน CPU ยังอาจใช้เวลาหลายชั่วโมง')
    print('=' * 72, flush=True)
    experiment(args)
    report = output / 'report.html'
    print(f'\nเสร็จแล้ว รายงานอยู่ที่: {report}')
    webbrowser.open(report.resolve().as_uri())
    input('\nกด Enter เพื่อปิดหน้าต่าง...')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nหยุดการเทรนตามคำสั่งผู้ใช้')
    except Exception:
        traceback.print_exc()
        input('\nเกิดข้อผิดพลาด กด Enter เพื่อปิด...')
        sys.exit(1)
