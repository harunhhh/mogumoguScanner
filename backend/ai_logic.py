import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_INPUT_SIZE = (224, 224)


class FoodAI:
    """料理画像からカロリーを推定するAIクラス。"""

    # 判定不能と見なす確信度の閾値（%）
    CONFIDENCE_THRESHOLD = 10.0

    # 1位と2位の差がこの値（%ポイント）未満なら判定不能扱いにする
    MARGIN_THRESHOLD = 3.0

    # TTA（Test-Time Augmentation）の有効化
    USE_TTA = True

    def __init__(self, model_path: str, csv_path: str):
        self.model_path = model_path
        self.csv_path = csv_path

        # クラス名の読み込み
        try:
            with open(os.path.join(_BASE_DIR, "classes.txt"), "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            print(f"[OK] {len(self.class_names)} 種類の料理データを読み込みました！")
        except FileNotFoundError:
            print("[ERROR] 'classes.txt' が見つかりません。同じフォルダに保存してください。")
            self.class_names = []

        self.model: Optional[tf.keras.Model] = self._load_model()
        self.df_calo = pd.read_csv(csv_path, encoding="utf-8")
        self.calorie_by_name = {
            row["食品名"]: row for _, row in self.df_calo.iterrows()
        }

        if self.model is not None and self.class_names:
            n_out = int(self.model.output_shape[-1])
            if n_out != len(self.class_names):
                print(
                    f"[WARN] モデルの出力数({n_out})とclasses.txtの件数({len(self.class_names)})が一致しません。"
                    f"先頭{len(self.class_names)}クラスのみを使用します。"
                )

    def _load_model(self) -> Optional[tf.keras.Model]:
        """Kerasモデルを読み込む。失敗時はNoneを返す。"""
        try:
            model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"[OK] モデルを読み込みました: {self.model_path}")
            return model
        except Exception as e:
            print(f"[ERROR] モデルの読み込みに失敗しました: {e}")
            return None

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """PIL画像を(224,224,3)のnumpy配列に変換する。"""
        # 学習時のtf.image.resizeに合わせてBILINEAR（PILの既定はBICUBIC）
        img = pil_image.resize(_INPUT_SIZE, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _tta_augmentations(self, pil_image: Image.Image) -> list:
        """TTA用の拡張画像リストを生成する（中央ズーム3段 × 水平反転の6パターン）。

        学習時のデータ拡張がRandomFlip(horizontal)とRandomZoomのみのため、
        それ以外の変換（明るさ・コントラスト・回転）はモデルにとって未知の分布になる。
        """
        w, h = pil_image.size
        variants = []
        for zoom in (1.0, 0.9, 0.8):
            cw, ch = int(w * zoom), int(h * zoom)
            left, top = (w - cw) // 2, (h - ch) // 2
            img = pil_image.crop((left, top, left + cw, top + ch))
            variants.append(img)
            variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        return variants

    def _mask_unused_classes(self, predictions: np.ndarray) -> np.ndarray:
        """classes.txtに載っていない余剰クラスを除外し、確率を再正規化する。

        学習時のデータセットに後から削除した追加クラスが含まれていたため、
        モデルの出力数がclasses.txtより多い。絶対に正解になり得ない出力を
        除いてから正規化することで、確信度が実態に合う。
        """
        n = len(self.class_names)
        if predictions.shape[1] <= n:
            return predictions
        trimmed = predictions[:, :n]
        return trimmed / np.maximum(trimmed.sum(axis=1, keepdims=True), 1e-12)

    def _get_top3(self, predictions: np.ndarray) -> list:
        """上位3候補を返す。"""
        top3_idx = np.argsort(predictions[0])[::-1][:3]
        return [
            {"name": self.class_names[idx], "confidence": float(predictions[0][idx] * 100)}
            for idx in top3_idx
        ]

    def _calorie_result(self, name: str, conf_score: float, top3: list) -> dict:
        row = self.calorie_by_name.get(name)
        if row is None:
            return {
                "name": name, "confidence": conf_score,
                "calories": "不明", "portion": "不明", "full_name": name,
                "top3": top3, "determined": True,
            }
        return {
            "name": name,
            "confidence": conf_score,
            "calories": row["エネルギー (kcal)"],
            "portion": row["目安量"],
            "full_name": row["食品名"],
            "top3": top3,
            "determined": True,
        }

    def predict(self, pil_image: Image.Image) -> dict:
        """PIL画像を受け取り、料理名・確信度・カロリーを返す。TTAで精度向上。"""
        if self.model is None or not self.class_names:
            return {
                "name": "モデル未ロード",
                "confidence": 0.0,
                "calories": "不明",
                "portion": "-",
                "full_name": "不明",
                "top3": [],
                "determined": False,
            }

        if self.USE_TTA:
            aug_images = self._tta_augmentations(pil_image)
            batch = np.stack([self._preprocess(img) for img in aug_images], axis=0)
            all_preds = self.model.predict(batch, verbose=0)
            predictions = np.mean(all_preds, axis=0, keepdims=True)
        else:
            img_array = np.expand_dims(self._preprocess(pil_image), axis=0)
            predictions = self.model.predict(img_array, verbose=0)

        predictions = self._mask_unused_classes(predictions)

        top3 = self._get_top3(predictions)
        conf_score = top3[0]["confidence"]
        margin = conf_score - top3[1]["confidence"] if len(top3) > 1 else conf_score

        # 確信度が低い、または1位と2位が拮抗している場合は「判定不能」（候補は返す）
        # determined を返すのは、確信度だけでは画面側が判定不能かを再現できないため。
        # 拮抗による判定不能は確信度が高いまま起きる
        if conf_score < self.CONFIDENCE_THRESHOLD or margin < self.MARGIN_THRESHOLD:
            return {
                "name": "判定不能（未登録）",
                "confidence": conf_score,
                "calories": "不明",
                "portion": "-",
                "full_name": "不明",
                "top3": top3,
                "determined": False,
            }

        return self._calorie_result(top3[0]["name"], conf_score, top3)
