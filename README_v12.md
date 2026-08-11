# ssu_unet_360 v12 — 합성 뷰 FBP를 입력으로 쓰는 denoiser

v11과의 유일한 차이는 **네트워크 입력**이다.

```
v11:  9뷰 sparse FBP                    → Pix2Pix/UNet → full-view 화질
v12:  9뷰 + 보간 네트워크 합성 뷰 FBP   → Pix2Pix/UNet → full-view 화질
```

합성 뷰는 [sinogram_interpolation_pix2pix_v9](/home/jwlee/sinogram_interpolation_pix2pix_v9)의
Δθ 조건화 보간 모델이 **재귀적 이등분**으로 만든다. 20° 간격이면 9 → 17 → 33뷰,
40° gap이 있는 시퀀스면 9 → 17 → 33 → 41뷰.

## 왜 하는가

보간 자체의 재구성 이득은 +2.5 dB 정도로 크지 않다. 진짜 이유는 **오차의 성질이 바뀌기
때문**이다.

| | 9뷰 | 합성 33/41뷰 |
|---|---|---|
| 배경 오차 | 화면을 가로지르는 **긴 coherent streak** | 짧고 고운 국소 texture |
| 물체 내부 | 완전 포화 | 크게 감소 |

CNN denoiser는 국소 texture는 잘 지우지만 화면을 가로지르는 streak은 잘 못 지운다.
v11이 지금 씨름하는 게 정확히 후자다. 따라서 **최종 산출물의 이득이 재구성 PSNR 차이보다
클 수 있다** — 이 프로젝트가 검증하려는 가설이 그것이다.

근거와 그림: [V10_NOTES.md](/home/jwlee/sinogram_interpolation_pix2pix_v9/V10_NOTES.md) §4.

## 실행 순서

```bash
# 0) (한 번) 보간 모델이 v12의 val/test subject를 학습하지 않았는지 보장
cd /home/jwlee/sinogram_interpolation_pix2pix_v9
python Step1_make_cache.py --split_from /home/jwlee/ssu_unet_360_v12/include/params.yml
python Step2_train_v10.py --model_type unet --num_down_G 6 \
    --exp_name v10_UNet_G6_splitv11 --gaps 40 20 10 \
    --samples_per_epoch 4000 --num_epochs 500 --batch_size 4 \
    --num_workers 8 --augment --gpu_id 0

# 1b) 합성 뷰 FBP 생성  -> datasets/{model_name}/fbp_interp/
cd /home/jwlee/ssu_unet_360_v12
python step1b_make_fbp_interp.py --target_gap 5

# 1c) 정규화 상수 재보정 -> 출력을 params.yml normalization 아래에 붙여넣기
python step1c_calibrate_norm.py

# 2~3) label(full-view FBP)은 v11 것을 그대로 쓴다 (step2, step3 불필요)

# 4) 데이터셋 구성 (input_name = fbp_interp)
python step4_Dataset_Fin.py --img_size 512

# 5) v11 가중치에서 fine-tune
python step5_Train_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 \
    --model_name Pix2Pix --gpu_id 0 \
    --init_from /home/jwlee/ssu_unet_360_v11/Pix2Pix/model_Gen3_7/ndG7_ndD4_dropout_3/ckpt/512/model_epoch1100.pth

# 6) 추론
python step6_Infer_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 --test_epoch <N>
```

## v11에서 바뀐 파일

| 파일 | 변경 |
|---|---|
| `include/params.yml` | `interp_project`, `interp_ckpt`, `input_name` 추가 |
| `step1b_make_fbp_interp.py` | **신규** — 합성 뷰 생성 + FBP |
| `step1c_calibrate_norm.py` | **신규** — 정규화 상수 재보정 |
| `step4_Dataset_Fin.py` | `input_name` 디렉터리를 읽고, 그에 맞는 정규화 상수 사용 |
| `step5_Train_FinModel.py` | `--init_from` (가중치만 로드), 실험 경로에 `input_name` 반영 |
| `step6_Infer_FinModel.py` | 실험 경로 접미사를 step5와 동기화 |
| `Pix2Pix/train.py` | `init_from` 지원 (`_load_weights_only`) |

`step1_make_fbp.py`, `step2`, `step3`은 손대지 않았다 — v11 baseline을 그대로 재현할 수
있어야 비교가 성립하기 때문이다.

## 세 가지 함정

**1. split 누수 — 이게 가장 중요하다.**
보간 모델의 합성 뷰가 denoiser의 *입력*이므로, denoiser의 test subject를 보간 모델이
학습했다면 그 입력은 이미 정답을 본 네트워크가 만든 것이다. 독립적으로 split하면
**v12 test 11개 중 9개가 보간 모델의 train에 들어간다.** 위 0) 단계가 이것을 맞추고,
`step1b`가 실행 전에 검증해 어긋나면 거부한다.

**2. 정규화 상수는 반드시 재보정해야 한다.**
`params.yml`의 `normalization.fbp`는 9뷰 FBP 분포에 맞춰 잡은 값이다. 합성 뷰 FBP는
streak이 줄어 강도 분포가 다르고, FBP 연산자의 `dbeta` 가중치도 뷰 수에 따라 달라진다
(9뷰×40° vs 41뷰×10°). 그대로 쓰면 입력이 엉뚱한 범위로 스케일되는데, **에러 없이
조용히 나쁜 결과만 나온다.** `step1c`가 train subject만으로 측정한다 (val/test로
상수를 정하면 그것도 누수다).

**3. baseline 비교는 `input_name`만 바꿔서 한다.**
`input_name: fbp_lh`로 두면 v11과 동일하게 동작한다. 같은 코드·같은 split·같은
하이퍼파라미터로 입력만 바꾼 A/B가 되어야 "합성 뷰 덕분"이라고 말할 수 있다.
실험 디렉터리에 `__fbp_interp` 접미사가 붙으므로 서로 덮어쓰지 않는다.
