# -*- coding: utf-8 -*-
"""
의공학 데이터 분석 및 시각화 과제
ECG(심전도) 심박 부정맥 분류 및 시각화 툴
- 데이터셋: MIT-BIH Arrhythmia (mitbih_train.csv / mitbih_test.csv)
           PTB Diagnostic ECG (ptbdb_normal.csv / ptbdb_abnormal.csv)
- 사용 라이브러리: numpy, pandas, matplotlib, seaborn (외부 머신러닝 라이브러리 미사용)
- 분류 알고리즘: 클래스별 평균 파형(centroid)과의 유클리드 거리를 이용한
                 최근접 중심점(Nearest Centroid) 분류기를 numpy로 직접 구현
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# 0. 경로 및 기본 설정
# -----------------------------------------------------------------
import argparse
import platform

_parser = argparse.ArgumentParser(description="ECG Nearest Centroid 분류·시각화 툴")
_parser.add_argument("--data", default="data",
                     help="mitbih_train/test.csv, ptbdb_normal/abnormal.csv가 있는 폴더 (기본: data)")
_parser.add_argument("--figures", default="figures", help="그림 저장 폴더 (기본: figures)")
_args = _parser.parse_args()

DATA_DIR = _args.data
FIG_DIR = _args.figures
os.makedirs(FIG_DIR, exist_ok=True)

N_BEAT_SAMPLES = 187  # 한 심박 신호의 시점(time step) 개수

# 한글 폰트 (OS별로 기본 설치된 폰트 사용)
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(
    platform.system(), "Noto Sans CJK JP")
plt.rcParams["axes.unicode_minus"] = False

MITBIH_CLASS_NAMES = {
    0: "N (정상박동)",
    1: "S (심방기원 조기박동)",
    2: "V (심실기원 조기박동)",
    3: "F (융합박동)",
    4: "Q (분류불가/페이싱박동)",
}


# -----------------------------------------------------------------
# 1. 데이터 로드 및 전처리 함수 (Pandas)
# -----------------------------------------------------------------
def load_beat_dataset(path):
    """
    심박 CSV(라벨 없는 마지막 열이 클래스)를 읽어
    (DataFrame, 특징행렬 X(numpy), 라벨벡터 y(numpy))를 반환한다.
    결측치가 있으면 0으로 채운다 (ECG 패딩 구간과 동일한 처리).
    """
    df = pd.read_csv(path, header=None)
    df.columns = [f"t{i}" for i in range(N_BEAT_SAMPLES)] + ["label"]

    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        df = df.fillna(0.0)
    print(f"  [load_beat_dataset] {os.path.basename(path)}: "
          f"shape={df.shape}, 결측치 보정={n_missing}개")

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    return df, X, y


def make_binary_dataset(normal_path, abnormal_path, test_ratio=0.2, seed=42):
    """
    PTBDB normal/abnormal 두 CSV를 합쳐 이진(0=정상,1=비정상) 데이터셋을 만들고,
    train/test로 분할한다. (사이킷런 없이 numpy로 직접 셔플/분할 구현)
    """
    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)

    # 두 파일 모두 마지막 열에 정상=0 / 비정상=1 라벨이 이미 포함되어 있음
    df = pd.concat([df_n, df_a], axis=0, ignore_index=True)
    df.columns = [f"t{i}" for i in range(N_BEAT_SAMPLES)] + ["label"]

    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        df = df.fillna(0.0)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_ratio)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx]), df


# -----------------------------------------------------------------
# 2. 전처리: 이동평균 기반 잡음 제거 (while문 직접 구현)
# -----------------------------------------------------------------
def smooth_signal(signal, window=5):
    """
    하나의 ECG 심박 신호(1차원 배열)에 단순 이동평균 필터를 적용해
    측정 잡음을 줄인다. while문으로 슬라이딩 윈도우를 직접 이동시킨다.
    """
    n = len(signal)
    smoothed = np.zeros(n)
    i = 0
    half = window // 2
    while i < n:
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.mean(signal[lo:hi])  # numpy 평균
        i += 1
    return smoothed


# -----------------------------------------------------------------
# 3. 최근접 중심점(Nearest Centroid) 분류기 직접 구현
# -----------------------------------------------------------------
def train_centroid_classifier(X, y):
    """
    클래스별 평균 파형(centroid)을 계산한다.
    -> {label: 평균벡터(numpy array)} 딕셔너리 반환
    """
    centroids = {}
    for label in np.unique(y):          # for문: 클래스 개수만큼 반복
        class_samples = X[y == label]   # 불리언 인덱싱 (numpy)
        centroids[label] = np.mean(class_samples, axis=0)
    return centroids


def predict_one(sample, centroids):
    """
    한 개의 심박 신호를 입력받아, 가장 가까운(유클리드 거리가 최소인)
    클래스 centroid의 라벨을 반환한다. if-else로 최소거리를 직접 비교.
    """
    best_label = None
    best_dist = np.inf
    for label, centroid in centroids.items():        # for문
        dist = np.sqrt(np.sum((sample - centroid) ** 2))  # 유클리드 거리
        if dist < best_dist:                          # if-else
            best_dist = dist
            best_label = label
    return best_label, best_dist


def predict_batch(X, centroids):
    """
    predict_one과 동일한 규칙을 전체 테스트셋에 벡터화하여 빠르게 적용한다.
    (다량의 데이터 처리를 위한 numpy 브로드캐스팅 활용)
    """
    labels = sorted(centroids.keys())
    C = np.stack([centroids[l] for l in labels])            # (n_class, n_feat)
    diff = X[:, None, :] - C[None, :, :]                    # (n_sample, n_class, n_feat)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))               # (n_sample, n_class)
    best_idx = np.argmin(dists, axis=1)
    preds = np.array([labels[i] for i in best_idx])
    return preds


def evaluate_classifier(y_true, y_pred, labels):
    """
    혼동행렬(confusion matrix), 전체 정확도, 클래스별 정확도(재현율)를 계산한다.
    """
    n_classes = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    conf = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):       # for문으로 한 건씩 혼동행렬에 누적
        conf[label_to_idx[t], label_to_idx[p]] += 1

    accuracy = np.trace(conf) / np.sum(conf)

    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):             # for문
        row_sum = conf[i].sum()
        if row_sum > 0:                    # if-else: 0으로 나누기 방지
            per_class_acc[i] = conf[i, i] / row_sum
        else:
            per_class_acc[i] = np.nan
    return conf, accuracy, per_class_acc


# -----------------------------------------------------------------
# 4. 시각화 함수들
# -----------------------------------------------------------------
def plot_class_mean_waveforms(df, class_names, save_path):
    """[시각화 1] Matplotlib - 클래스별 평균 ECG 파형 비교"""
    means = df.groupby("label").mean()  # pandas groupby
    plt.figure(figsize=(9, 5))
    for label in means.index:
        plt.plot(means.loc[label].values, label=class_names.get(int(label), str(label)))
    plt.title("MIT-BIH 클래스별 평균 ECG 심박 파형")
    plt.xlabel("Time step (sample index)")
    plt.ylabel("Normalized amplitude")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


def plot_class_distribution(train_y, test_y, class_names, save_path):
    """[시각화 2] Matplotlib - train/test 클래스 분포 비교 (그룹 bar)"""
    labels = sorted(class_names.keys())
    train_counts = [np.sum(train_y == l) for l in labels]
    test_counts = [np.sum(test_y == l) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, train_counts, width, label="Train")
    plt.bar(x + width / 2, test_counts, width, label="Test")
    plt.xticks(x, [class_names[l] for l in labels], rotation=20, ha="right", fontsize=8)
    plt.ylabel("심박 개수")
    plt.title("MIT-BIH 클래스별 데이터 분포 (불균형 확인)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


def plot_denoising_example(raw, smoothed, save_path):
    """[시각화 3] Matplotlib - 원본 vs 이동평균 잡음제거 신호 비교"""
    plt.figure(figsize=(9, 4))
    plt.plot(raw, label="원본 신호", alpha=0.6)
    plt.plot(smoothed, label="이동평균 적용", linewidth=2)
    plt.title("이동평균 필터를 이용한 ECG 잡음 제거 예시")
    plt.xlabel("Time step")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


def plot_confusion_matrix(conf, class_names, labels, title, save_path):
    """[시각화 4] Seaborn - 혼동행렬 heatmap"""
    tick_labels = [class_names.get(l, str(l)) for l in labels]
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("예측(Predicted) 클래스")
    plt.ylabel("실제(True) 클래스")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


def plot_signal_energy_boxplot(df, class_names, save_path):
    """[시각화 5] Seaborn - 클래스별 심박 신호 에너지(sum of squares) 분포 boxplot
    (개별 심박이 0~1로 정규화되어 있어 최대진폭은 거의 항상 1이 되므로,
     QRS파형의 강도를 더 잘 반영하는 신호 에너지를 대신 사용)"""
    df_plot = df.copy()
    beat_values = df_plot.iloc[:, :N_BEAT_SAMPLES].to_numpy()
    df_plot["signal_energy"] = np.sum(beat_values ** 2, axis=1)  # numpy 기반 행별 에너지 계산
    df_plot["class_name"] = df_plot["label"].map(lambda l: class_names.get(int(l), str(l)))

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_plot, x="class_name", y="signal_energy")
    plt.title("클래스별 ECG 심박 신호 에너지 분포")
    plt.xlabel("부정맥 클래스")
    plt.ylabel("Signal energy (sum of squares)")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


def plot_binary_mean_waveform(df, save_path):
    """[시각화 6] Matplotlib - PTBDB 정상 vs 비정상 평균 파형 비교"""
    means = df.groupby("label").mean()
    plt.figure(figsize=(9, 5))
    name_map = {0: "정상(Normal)", 1: "비정상(Abnormal)"}
    for label in means.index:
        plt.plot(means.loc[label].values, label=name_map.get(int(label), str(label)))
    plt.title("PTBDB 정상 vs 비정상 평균 ECG 파형 비교")
    plt.xlabel("Time step")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장됨: {save_path}")


# -----------------------------------------------------------------
# 5. 메인 실행부
# -----------------------------------------------------------------
def main():
    print("=" * 70)
    print("Part 1. MIT-BIH 5클래스 부정맥 분류")
    print("=" * 70)

    train_df, X_train, y_train = load_beat_dataset(os.path.join(DATA_DIR, "mitbih_train.csv"))
    test_df, X_test, y_test = load_beat_dataset(os.path.join(DATA_DIR, "mitbih_test.csv"))

    # --- 기본 통계 (numpy + pandas) ---
    print("\n[학습 데이터 클래스별 통계]")
    stats = train_df.groupby("label").agg(
        count=("t0", "size"),
        mean_amp=("t0", "mean"),
    )
    print(stats)

    overall_mean = np.mean(X_train)
    overall_std = np.std(X_train)
    print(f"\n전체 진폭 평균={overall_mean:.4f}, 표준편차={overall_std:.4f}")

    # --- 잡음제거 예시 (while문 함수) ---
    example_raw = X_train[0]
    example_smoothed = smooth_signal(example_raw, window=7)
    plot_denoising_example(example_raw, example_smoothed,
                            os.path.join(FIG_DIR, "fig3_denoising_example.png"))

    # --- 분류기 학습 ---
    centroids = train_centroid_classifier(X_train, y_train)

    # predict_one 동작 예시 (알고리즘 설명용)
    print("\n[predict_one 동작 예시 - 테스트 샘플 5개]")
    for i in range(5):                       # for문
        label, dist = predict_one(X_test[i], centroids)
        print(f"  sample {i}: 실제={y_test[i]}, 예측={label}, 최소거리={dist:.3f}")

    # --- 전체 테스트셋 평가 ---
    y_pred = predict_batch(X_test, centroids)
    labels = sorted(MITBIH_CLASS_NAMES.keys())
    conf, acc, per_class_acc = evaluate_classifier(y_test, y_pred, labels)

    print(f"\n[MIT-BIH 분류 결과] 전체 정확도 = {acc * 100:.2f}%")
    for lab, pacc in zip(labels, per_class_acc):
        print(f"  클래스 {lab} ({MITBIH_CLASS_NAMES[lab]}): 재현율 = {pacc * 100:.2f}%")

    # --- 시각화 ---
    plot_class_mean_waveforms(train_df, MITBIH_CLASS_NAMES,
                               os.path.join(FIG_DIR, "fig1_class_mean_waveforms.png"))
    plot_class_distribution(y_train, y_test, MITBIH_CLASS_NAMES,
                             os.path.join(FIG_DIR, "fig2_class_distribution.png"))
    plot_confusion_matrix(conf, MITBIH_CLASS_NAMES, labels,
                           "MIT-BIH 최근접 중심점 분류기 혼동행렬",
                           os.path.join(FIG_DIR, "fig4_confusion_matrix_mitbih.png"))
    plot_signal_energy_boxplot(train_df, MITBIH_CLASS_NAMES,
                                os.path.join(FIG_DIR, "fig5_signal_energy_boxplot.png"))

    print("\n" + "=" * 70)
    print("Part 2. PTBDB 정상/비정상 이진 분류 (추가 분석)")
    print("=" * 70)

    (Xb_train, yb_train), (Xb_test, yb_test), ptb_df = make_binary_dataset(
        os.path.join(DATA_DIR, "ptbdb_normal.csv"),
        os.path.join(DATA_DIR, "ptbdb_abnormal.csv"),
    )
    print(f"  PTBDB 전체 {len(ptb_df)}건 -> train {len(yb_train)} / test {len(yb_test)}")
    print(ptb_df["label"].value_counts().rename({0: "정상", 1: "비정상"}))

    bin_centroids = train_centroid_classifier(Xb_train, yb_train)
    yb_pred = predict_batch(Xb_test, bin_centroids)
    bin_labels = [0, 1]
    bin_conf, bin_acc, bin_per_class_acc = evaluate_classifier(yb_test, yb_pred, bin_labels)

    print(f"\n[PTBDB 이진분류 결과] 전체 정확도 = {bin_acc * 100:.2f}%")
    print(f"  정상 재현율 = {bin_per_class_acc[0] * 100:.2f}%, "
          f"비정상 재현율 = {bin_per_class_acc[1] * 100:.2f}%")

    plot_binary_mean_waveform(ptb_df, os.path.join(FIG_DIR, "fig6_ptbdb_mean_waveform.png"))
    plot_confusion_matrix(bin_conf, {0: "정상", 1: "비정상"}, bin_labels,
                           "PTBDB 최근접 중심점 분류기 혼동행렬",
                           os.path.join(FIG_DIR, "fig7_confusion_matrix_ptbdb.png"))

    print("\n모든 분석 및 시각화가 완료되었습니다. figures/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
