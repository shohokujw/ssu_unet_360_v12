#!/usr/bin/env python3
"""학습 상태 한 눈에 보기.

stdout(nohup.out)에는 tqdm 진행바밖에 없고 loss는 TensorBoard event 파일로만
가므로, 두 곳을 함께 읽어 요약한다.

    python3 check_train.py
    python3 check_train.py --log nohup.out --watch 30   # 30초마다 갱신
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

current_folder = os.path.dirname(os.path.realpath(__file__))

SCRIPT = 'step5_Train_FinModel.py'
# 'Epoch 3/1300:  45%|## | 437/969 [01:03<03:36,  3.46it/s]'
PROGRESS = re.compile(
    r'Epoch (\d+)/(\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<')


def fmt(seconds):
    """초를 '3일 4시간 5분' 꼴로."""
    if seconds is None or seconds != seconds or seconds < 0:
        return '?'
    seconds = int(seconds)
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d:
        return f'{d}일 {h}시간 {m}분'
    if h:
        return f'{h}시간 {m}분'
    return f'{m}분 {r % 60}초'


def find_pid():
    try:
        out = subprocess.run(['pgrep', '-f', SCRIPT], capture_output=True,
                             text=True).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def elapsed_of(pid):
    """프로세스가 살아있는 시간(초)."""
    try:
        out = subprocess.run(['ps', '-o', 'etimes=', '-p', str(pid)],
                             capture_output=True, text=True).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def last_progress(log_path):
    """로그 끝에서 마지막 진행바를 파싱."""
    try:
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode('utf-8', 'replace')
    except FileNotFoundError:
        return None
    hits = PROGRESS.findall(tail.replace('\r', '\n'))
    if not hits:
        return None
    ep, total_ep, pct, it, total_it, ep_elapsed = hits[-1]
    return dict(epoch=int(ep), total_epoch=int(total_ep), pct=int(pct),
                it=int(it), total_it=int(total_it), ep_elapsed=ep_elapsed)


def read_losses(log_dir):
    """TensorBoard event 파일에서 스칼라 최신값."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None, 'tensorboard 미설치'
    if not os.path.isdir(log_dir):
        return None, f'로그 디렉터리 없음: {log_dir}'
    ea = event_accumulator.EventAccumulator(
        log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        return None, '아직 기록된 loss 없음 (첫 epoch 완료 전)'
    return {t: ea.Scalars(t)[-1] for t in tags}, None


def report(args):
    log_path = args.log if os.path.isabs(args.log) else os.path.join(
        current_folder, args.log)

    pid = find_pid()
    prog = last_progress(log_path)

    print('=' * 62)
    if pid:
        el = elapsed_of(pid)
        print(f'상태      : 실행 중  (PID {pid}, 경과 {fmt(el)})')
    else:
        el = None
        print('상태      : ✗ 실행 중인 프로세스 없음')
    print(f'로그      : {log_path}')

    if not prog:
        print('진행      : 진행바를 찾지 못했습니다 (아직 시작 전이거나 로그 형식이 다름)')
        print('=' * 62)
        return

    ep, tot = prog['epoch'], prog['total_epoch']
    # 완료한 epoch 수(소수 포함). 현재 epoch은 진행률만큼만 센다.
    done = (ep - 1) + prog['it'] / prog['total_it']
    print(f"진행      : epoch {ep} / {tot}   "
          f"(전체 {done / tot * 100:.1f}%)")
    print(f"현재 epoch: {prog['pct']}%  "
          f"({prog['it']}/{prog['total_it']} it, {prog['ep_elapsed']} 경과)")

    if el and done > 0:
        per_epoch = el / done
        remain = (tot - done) * per_epoch
        eta = datetime.now() + timedelta(seconds=remain)
        print(f'epoch당   : {fmt(per_epoch)}')
        print(f'남은 시간 : 약 {fmt(remain)}  (완료 예정 {eta:%m/%d %H:%M})')

    losses, err = read_losses(args.logdir or default_logdir())
    print('-' * 62)
    if err:
        print(f'loss      : {err}')
    else:
        recorded = max(s.step for s in losses.values())
        print(f'loss      : (마지막 기록 epoch {recorded})')
        for tag in sorted(losses):
            print(f'  {tag:<22} {losses[tag].value:>12.4f}')

    ckpt_dir = args.ckptdir or default_ckptdir()
    print('-' * 62)
    if os.path.isdir(ckpt_dir):
        cks = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pth'))
        if cks:
            print(f'체크포인트: {len(cks)}개, 최신 {cks[-1]}')
        else:
            print(f'체크포인트: 아직 없음  ({ckpt_dir})')
    else:
        print(f'체크포인트: 디렉터리 없음  ({ckpt_dir})')
    print('=' * 62)


def _exp_root():
    """params.yml에서 현재 실험 경로를 유추."""
    sys.path.insert(0, current_folder)
    from config_utils import load_ct_params
    p = load_ct_params()
    data_name = p['model_name']
    input_name = p.get('input_name', 'fbp_lh')
    suffix = '' if input_name == 'fbp_lh' else f'__{input_name}'
    exp = f'ndG7_ndD4_dropout_3{suffix}'
    return os.path.join(current_folder, 'Pix2Pix', data_name, exp)


def default_logdir():
    return os.path.join(_exp_root(), 'log', '512')


def default_ckptdir():
    return os.path.join(_exp_root(), 'ckpt', '512')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='nohup.out', help='stdout 로그 파일')
    ap.add_argument('--logdir', default=None, help='TensorBoard 로그 디렉터리')
    ap.add_argument('--ckptdir', default=None, help='체크포인트 디렉터리')
    ap.add_argument('--watch', type=int, default=0,
                    help='N초마다 갱신 (0이면 한 번만)')
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                os.system('clear')
                report(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        report(args)
