"""UECFOOD100 で MobileNetV2 をファインチューニングする学習スクリプト。

使い方:
    python train.py --data-dir ~/datasets/UECFOOD100

UECFOOD100 の画像は定食のトレーなど複数の料理が写っているものが多く、
同じ画像ファイルが複数のクラスフォルダに重複して置かれている。
フォルダ単位で読むと同一画像に別々の正解ラベルが付くため、各フォルダの
bb_info.txt にある矩形で該当料理を切り出してから学習する。

学習後、backend/food_model.keras と backend/class_ids.txt を更新する。
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMAGE_SIZE = (224, 224)
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_samples(data_dir, class_ids, use_crop):
    """(画像パス, 矩形, クラス番号, 元画像ID) のリストを作る。

    use_crop=False のときは画像全体を矩形として扱う。従来のフォルダ丸ごと
    学習を再現して精度を比較するための逃げ道。
    """
    samples = []
    missing = 0
    for class_index, class_id in enumerate(class_ids):
        folder = os.path.join(data_dir, class_id)
        bb_path = os.path.join(folder, "bb_info.txt")

        if use_crop and os.path.exists(bb_path):
            with open(bb_path) as f:
                next(f)  # ヘッダ行 "img x1 y1 x2 y2"
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    img_id = parts[0]
                    x1, y1, x2, y2 = (int(v) for v in parts[1:])
                    path = os.path.join(folder, f"{img_id}.jpg")
                    if not os.path.exists(path):
                        missing += 1
                        continue
                    samples.append((path, (x1, y1, x2, y2), class_index, img_id))
        else:
            for name in os.listdir(folder):
                if not name.lower().endswith(".jpg"):
                    continue
                img_id = os.path.splitext(name)[0]
                # 0,0,0,0 は「矩形指定なし＝画像全体」の目印として扱う
                samples.append((os.path.join(folder, name), (0, 0, 0, 0), class_index, img_id))

    if missing:
        print(f"[警告] bb_info.txt に載っているが実体が無い画像を {missing} 件スキップしました")
    return samples


def add_full_image_samples(samples):
    """1皿しか写っていない画像に限り、画像全体も学習データに加える。

    アプリでは料理が小さく写った写真も来るので、切り出しだけで学習すると
    構図の違いで崩れる。ただし複数クラスに登場する画像を全体像のまま入れると、
    同じ画像に別々の正解が付く元の不具合が復活するので対象から外す。
    """
    classes_of = defaultdict(set)
    boxes_of = defaultdict(int)
    for _, _, class_index, img_id in samples:
        classes_of[img_id].add(class_index)
        boxes_of[(class_index, img_id)] += 1

    extra = []
    seen = set()
    for path, _, class_index, img_id in samples:
        if img_id in seen:
            continue
        if len(classes_of[img_id]) == 1 and boxes_of[(class_index, img_id)] == 1:
            seen.add(img_id)
            # (0,0,0,0) は「矩形指定なし＝画像全体」の目印
            extra.append((path, (0, 0, 0, 0), class_index, img_id))
    return samples + extra


def split_samples(samples, val_ratio):
    """元画像単位で訓練用と検証用に分ける。

    同じ写真から切り出した矩形が訓練側と検証側に分かれると、検証データに
    実質的に見たことのある画像が混ざり、精度を過大評価してしまう。
    """
    by_image = defaultdict(list)
    for s in samples:
        by_image[s[3]].append(s)

    image_ids = sorted(by_image)
    rng = np.random.default_rng(SEED)
    rng.shuffle(image_ids)

    n_val = int(len(image_ids) * val_ratio)
    val_ids = set(image_ids[:n_val])

    train = [s for i in image_ids if i not in val_ids for s in by_image[i]]
    val = [s for i in image_ids if i in val_ids for s in by_image[i]]
    return train, val


def decode_and_crop(path, box, margin_lo, margin_hi):
    """画像を読み込み、矩形にマージンを付けて切り出し、224x224 に揃える。

    margin_hi > margin_lo のときは余白を毎回ランダムに選ぶ。切り出しだけで
    学習するとモデルが「1皿が画面いっぱい」の構図しか知らなくなり、
    実際のアプリでトレー全体を撮られたときに崩れるため、寄り引きを散らす。
    """
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    shape = tf.shape(img)
    height, width = shape[0], shape[1]

    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    margin = tf.cond(
        tf.greater(margin_hi, margin_lo),
        lambda: tf.random.uniform([], margin_lo, margin_hi),
        lambda: tf.constant(margin_lo, tf.float32),
    )

    # 全要素 0 なら矩形指定なしとみなして画像全体を使う
    def full_image():
        return 0, 0, width, height

    def cropped():
        # 料理の縁が切れると判別しづらくなるので少しだけ外側を含める
        mx = tf.cast(tf.cast(x2 - x1, tf.float32) * margin, tf.int32)
        my = tf.cast(tf.cast(y2 - y1, tf.float32) * margin, tf.int32)
        cx1 = tf.maximum(x1 - mx, 0)
        cy1 = tf.maximum(y1 - my, 0)
        cx2 = tf.minimum(x2 + mx, width)
        cy2 = tf.minimum(y2 + my, height)
        # 座標が壊れている行があっても落ちないように最低 1px は確保する
        return cx1, cy1, tf.maximum(cx2, cx1 + 1), tf.maximum(cy2, cy1 + 1)

    cx1, cy1, cx2, cy2 = tf.cond(
        tf.reduce_all(tf.equal(box, 0)),
        lambda: tuple(tf.cast(v, tf.int32) for v in full_image()),
        cropped,
    )

    img = img[cy1:cy2, cx1:cx2]
    img = tf.image.resize(img, IMAGE_SIZE)
    # MobileNetV2 が必要とする -1〜1 への変換はモデル内部で行う
    return img / 255.0


def build_dataset(samples, num_classes, batch_size, margin_lo, margin_hi, training):
    paths = tf.constant([s[0] for s in samples])
    boxes = tf.constant(np.array([s[1] for s in samples], dtype=np.int32))
    labels = tf.constant(np.array([s[2] for s in samples], dtype=np.int32))

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, labels))
    if training:
        ds = ds.shuffle(len(samples), seed=SEED, reshuffle_each_iteration=True)
    else:
        # 検証は毎回同じ切り出しになるようランダム化しない
        margin_hi = margin_lo
    ds = ds.map(
        lambda p, b, l: (
            decode_and_crop(p, b, margin_lo, margin_hi),
            tf.one_hot(l, num_classes),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def compute_class_weights(samples, num_classes):
    """枚数が 7 倍以上ばらついているので、少ないクラスの損失を重くする。"""
    counts = np.bincount([s[2] for s in samples], minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1  # 0 除算よけ
    weights = counts.sum() / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3)),
            augment,
            # 学習済み重みは -1〜1 の入力を前提にしている
            layers.Rescaling(scale=2.0, offset=-1.0),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


def evaluate(model, val_ds, class_ids, class_names):
    probs = model.predict(val_ds, verbose=0)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])
    y_pred = probs.argmax(axis=1)

    top1 = float((y_pred == y_true).mean())
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3 = float(np.mean([t in row for t, row in zip(y_true, top3_idx)]))
    print(f"\n検証データ Top-1 正解率: {top1:.4f}")
    print(f"検証データ Top-3 正解率: {top3:.4f}")

    per_class = [
        (class_names[c], float((y_pred[y_true == c] == c).mean()), int((y_true == c).sum()))
        for c in range(len(class_ids))
        if (y_true == c).sum() > 0
    ]
    per_class.sort(key=lambda r: r[1])
    print("\n正解率が低いクラス ワースト10 (料理名, 正解率, 検証枚数):")
    for name, acc, n in per_class[:10]:
        print(f"  {name}: {acc:.3f} ({n}枚)")
    return top1, top3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="クラスIDごとのフォルダを含むデータセットのパス")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=10, help="第1段階（ヘッドのみ）のエポック数")
    p.add_argument("--finetune-epochs", type=int, default=50, help="第2段階のエポック数")
    p.add_argument("--fine-tune-at", type=int, default=100, help="この層より後ろを解凍する")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--crop-margin", type=float, default=0.08, help="矩形の外側に含める余白の割合")
    p.add_argument("--crop-margin-max", type=float, default=None,
                   help="指定すると学習時の余白を --crop-margin との間でランダムに選ぶ")
    p.add_argument("--include-full", action="store_true",
                   help="1皿しか写っていない画像は画像全体も学習データに加える")
    p.add_argument("--no-crop", action="store_true",
                   help="bb_info.txt を使わず画像全体で学習する（従来手法との比較用）")
    p.add_argument("--no-class-weight", action="store_true", help="クラス重み付けを無効にする")
    p.add_argument("--tag", default="",
                   help="出力ファイル名に付ける接尾辞。比較用の学習で本番モデルを上書きしないために使う")
    args = p.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} / GPU: {gpus or 'なし（CPUで学習します）'}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data_dir = os.path.expanduser(args.data_dir)

    # 数字以外の名前のフォルダ（.ipynb_checkpoints など）は key=int が落ちるので除外する
    class_ids = sorted(
        (d for d in os.listdir(data_dir)
         if d.isdigit() and os.path.isdir(os.path.join(data_dir, d))),
        key=int,
    )
    print(f"{len(class_ids)} クラスを検出しました: {class_ids[:5]} ... {class_ids[-5:]}")

    class_names_path = os.path.join(_BASE_DIR, "backend", "classes.txt")
    with open(class_names_path, encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    if len(class_names) != len(class_ids):
        raise SystemExit(
            f"classes.txt の件数({len(class_names)})とデータセットのクラス数({len(class_ids)})が"
            f"一致しません。学習前に揃えてください。"
        )

    use_crop = not args.no_crop
    samples = load_samples(data_dir, class_ids, use_crop)
    if use_crop and args.include_full:
        before = len(samples)
        samples = add_full_image_samples(samples)
        print(f"1皿だけ写っている画像の全体像を {len(samples) - before} 件追加しました")

    train_s, val_s = split_samples(samples, args.val_ratio)
    print(
        f"{'bb_info.txt の矩形で切り出し' if use_crop else '画像全体（切り出しなし）'}: "
        f"学習 {len(train_s)} 件 / 検証 {len(val_s)} 件"
    )

    num_classes = len(class_ids)
    margin_hi = args.crop_margin if args.crop_margin_max is None else args.crop_margin_max
    if margin_hi > args.crop_margin:
        print(f"学習時の余白を {args.crop_margin:.2f}〜{margin_hi:.2f} でランダム化します")
    train_ds = build_dataset(
        train_s, num_classes, args.batch_size, args.crop_margin, margin_hi, True
    )
    val_ds = build_dataset(
        val_s, num_classes, args.batch_size, args.crop_margin, args.crop_margin, False
    )

    class_weight = None if args.no_class_weight else compute_class_weights(train_s, num_classes)

    model, base = build_model(num_classes)

    # ランダム初期化の Dense をいきなり学習済み層につなぐと、最初の大きな勾配で
    # 学習済み重みが壊れる。先にヘッドだけ慣らしてから解凍する
    print("\n=== 第1段階: ヘッドのみ学習 ===")
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
              class_weight=class_weight)

    print("\n=== 第2段階: ファインチューニング ===")
    base.trainable = True
    for layer in base.layers[: args.fine_tune_at]:
        layer.trainable = False
    # BatchNormalization は解凍しない。解凍すると移動平均が小さなバッチで
    # 上書きされ、学習時と推論時で挙動が食い違って精度が落ちる
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 1e-5)
    model.summary()

    best_path = os.path.join(_BASE_DIR, "backend", f"food_model_best{args.tag}.keras")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        ],
    )

    evaluate(model, val_ds, class_ids, class_names)

    model_name = f"food_model{args.tag}.keras"
    ids_name = f"class_ids{args.tag}.txt"
    model.save(os.path.join(_BASE_DIR, "backend", model_name))
    # 学習に使ったクラスIDの並びも必ず残す。これが無かったために
    # モデルの出力数と classes.txt の件数がずれたまま気づけない状態になっていた
    with open(os.path.join(_BASE_DIR, "backend", ids_name), "w", encoding="utf-8") as f:
        f.write("\n".join(class_ids))
    print(f"\nbackend/{model_name} と backend/{ids_name} を更新しました。")


if __name__ == "__main__":
    main()
