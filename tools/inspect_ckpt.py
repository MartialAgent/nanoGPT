"""ckpt.pt / *.bin / meta.pkl 안을 들여다보는 뷰어.

VSCode 확장으로는 이 세 포맷이 제대로 안 보인다.
- *.bin  : 헤더 없는 uint16 배열이라 hex 뷰어로는 바이트만 보임
- ckpt.pt: traced 모델이 아니라 state_dict + 메타 dict라 Netron이 그래프를 못 그림
- meta.pkl: 파이썬 pickle이라 텍스트 에디터로 열면 깨짐

사용법 (nanoGPT/ 에서 실행):
    python tools/inspect_ckpt.py                                  # 아는 파일 전부 스캔
    python tools/inspect_ckpt.py data/shakespeare_char/meta.pkl
    python tools/inspect_ckpt.py data/shakespeare_char/train.bin --head 300
    python tools/inspect_ckpt.py out-shakespeare-char/ckpt.pt --all
"""

import argparse
import os
import pickle
import sys

# Windows 콘솔(cp949)에서 셰익스피어 텍스트가 깨지지 않게
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGETS = [
    "data/shakespeare_char/meta.pkl",
    "data/shakespeare_char/train.bin",
    "data/shakespeare_char/val.bin",
    "out-shakespeare-char/ckpt.pt",
]


def hr(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def find_meta(path):
    """*.bin 옆의 meta.pkl 을 찾아 토큰->문자 매핑을 돌려준다 (없으면 None)."""
    meta_path = os.path.join(os.path.dirname(path), "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def show_pkl(path, args):
    hr(f"PKL  {path}  ({human(os.path.getsize(path))})")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"type: {type(obj).__name__}")
    if not isinstance(obj, dict):
        print(repr(obj)[: args.head * 4])
        return

    for k, v in obj.items():
        if isinstance(v, dict):
            items = list(v.items())
            shown = items if args.all else items[: args.head]
            print(f"\n{k}: dict(len={len(v)})")
            print("  " + ", ".join(f"{ki!r}->{vi!r}" for ki, vi in shown))
            if len(items) > len(shown):
                print(f"  ... (+{len(items) - len(shown)} more, --all 로 전체 출력)")
        else:
            print(f"\n{k}: {v!r}")


def show_bin(path, args):
    import numpy as np

    size = os.path.getsize(path)
    hr(f"BIN  {path}  ({human(size)})")

    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"dtype    : uint16 (little-endian, 헤더 없음)")
    print(f"tokens   : {len(data):,}  ({size:,} bytes / 2)")
    print(f"min/max  : {int(data.min())} / {int(data.max())}")

    n = len(data) if args.all else min(args.head, len(data))
    print(f"\n첫 {n} 토큰 ID:")
    print(" ".join(str(int(t)) for t in data[:n]))

    meta = find_meta(path)
    if meta is None or "itos" not in meta:
        print("\n(옆에 meta.pkl 이 없어 문자로 복원하지 못함)")
        return
    itos = meta["itos"]
    print(f"\nmeta.pkl 의 itos 로 복원한 첫 {n} 글자:")
    print("-" * 70)
    print("".join(itos.get(int(t), "\ufffd") for t in data[:n]))
    print("-" * 70)


def show_pt(path, args):
    import torch

    hr(f"CKPT {path}  ({human(os.path.getsize(path))})")

    # torch 2.5: weights_only=True 가 안전하지만 config dict 때문에 실패할 수 있어 폴백
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"예상과 다른 형식: {type(ckpt)}")
        return

    print(f"top-level keys: {list(ckpt.keys())}")
    for key in ("iter_num", "best_val_loss", "model_args", "config"):
        if key in ckpt:
            v = ckpt[key]
            if isinstance(v, dict):
                print(f"\n{key}:")
                for k2, v2 in v.items():
                    print(f"  {k2:24} = {v2!r}")
            else:
                print(f"\n{key}: {v!r}")

    sd = ckpt.get("model")
    if sd is None:
        return

    total = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
    print(f"\nmodel state_dict: {len(sd)} tensors, {total:,} params ({total / 1e6:.2f}M)")
    items = list(sd.items())
    shown = items if args.all else items[: args.head]
    print(f"{'tensor':<48} {'shape':<20} dtype")
    print("-" * 84)
    for k, v in shown:
        shape = tuple(v.shape) if hasattr(v, "shape") else "-"
        dtype = str(v.dtype).replace("torch.", "") if hasattr(v, "dtype") else "-"
        print(f"{k:<48} {str(shape):<20} {dtype}")
    if len(items) > len(shown):
        print(f"... (+{len(items) - len(shown)} more, --all 로 전체 출력)")

    # 옵티마이저 상태까지 들어있으면 체크포인트가 왜 큰지 알려준다
    if "optimizer" in ckpt:
        print("\n'optimizer' 키 있음 -> AdamW 의 exp_avg/exp_avg_sq 때문에 파일이 파라미터의 약 3배")


DISPATCH = {".pkl": show_pkl, ".bin": show_bin, ".pt": show_pt, ".pth": show_pt}


def inspect(path, args):
    if not os.path.exists(path):
        print(f"[건너뜀] 파일 없음: {path}")
        return
    fn = DISPATCH.get(os.path.splitext(path)[1].lower())
    if fn is None:
        print(f"[건너뜀] 지원하지 않는 확장자: {path}")
        return
    fn(path, args)


def main():
    p = argparse.ArgumentParser(description="nanoGPT 의 ckpt.pt / *.bin / meta.pkl 내용 보기")
    p.add_argument("paths", nargs="*", help="비워두면 알려진 파일을 전부 스캔")
    p.add_argument("--head", type=int, default=20, help="항목/토큰을 몇 개까지 볼지 (기본 20)")
    p.add_argument("--all", action="store_true", help="자르지 않고 전부 출력")
    args = p.parse_args()

    targets = args.paths or [os.path.join(ROOT, t) for t in DEFAULT_TARGETS]
    for t in targets:
        inspect(t, args)


if __name__ == "__main__":
    main()
