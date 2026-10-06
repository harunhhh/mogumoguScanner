# 🍛 もぐもぐスキャナー

料理の写真を撮るだけで、**AIが料理名を判定しカロリーを推定**する Web アプリです。

---

## 📸 デモ

| 入力 | 出力 |
|------|------|
| カメラ撮影 or 画像アップロード | 料理名・確信度・推定カロリー (kcal) |

**公開URL**: https://mogumogu-mauve.vercel.app

---

## 🧠 技術スタック

| レイヤー | 技術 |
|----------|------|
| **フロントエンド** | [Next.js](https://nextjs.org/)（React / TypeScript / Tailwind CSS） |
| **バックエンド API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **AIモデル** | CNN（MobileNetV2 ベースのファインチューニング） |
| **学習フレームワーク** | TensorFlow / Keras |
| **学習環境** | Google Colab |
| **データセット** | [UECFOOD100](https://www.kaggle.com/datasets/rkuo2000/uecfood100?resource=download)（100種） |
| **カロリーDB** | 自作 CSV（100品目） |

---

## 🔍 技術選定の理由

### Next.js + FastAPI
当初は Streamlit で UI を組んでいたが、独自デザイン（背景ぼかし・カード UI・グラデーションボタンなど）を作り込むにつれて Streamlit のコンポーネント制約と CSS のせめぎ合いが大きくなったため、React ベースの Next.js に移行。
AI 推論部分は FastAPI の `/predict` エンドポイントとして切り出し、フロントエンドとバックエンドを分離した構成にした。

### MobileNetV2（転移学習）
料理画像の分類には画像認識に特化した CNN が必要だが、ゼロから学習させるにはデータ量・計算コストともに現実的でない。
MobileNetV2 は ImageNet で事前学習済みのモデルの中でも軽量かつ精度が高く、Google Colab の無料 GPU 環境でも現実的な時間でファインチューニングが完了する。
また `.keras` 形式での保存・読み込みが Keras と自然に統合されており、デプロイ時の扱いやすさも選定理由のひとつ。

### TensorFlow / Keras
Python の機械学習フレームワークとして最も情報量が多く、トラブル時に参照できるドキュメント・事例が豊富。
Keras の高レベル API により、モデル構築・コンパイル・学習の記述がシンプルに書けるため、学習コードの可読性を保てた。

### Google Colab
GPU を無料で利用できる環境として採用。
MobileNetV2 のファインチューニングには GPU が実質必須であり、ローカル環境に依存せずにどこでも学習を再実行できる点も利点。
Google Drive との連携でデータセットの管理・モデルの保存も容易に行える。

### UECFOOD100
日本食を中心とした 100 クラスの食品画像データセット。
国内での利用を想定したアプリのため、和食の認識精度が高いデータセットを優先して選定した。

## 🤖 AIモデルの仕組み

### アーキテクチャ：CNN（畳み込みニューラルネットワーク）

```
入力画像 (224×224 px)
    ↓
データ拡張（RandomFlip / Rotation / Zoom / Contrast）
    ↓
MobileNetV2（ImageNet 事前学習済み）
  └─ 最初の 100 層はフリーズ
  └─ 100 層目以降をファインチューニング
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.6) + L2正則化
    ↓
Dense → Softmax（100クラス分類）
```

### 学習の工夫
- **転移学習**：ImageNet 学習済みの MobileNetV2 をベースに利用
- **ファインチューニング**：上位層のみ再学習し過学習を防止
- **早期停止 (EarlyStopping)**：val_accuracy が 5 エポック改善しなければ自動停止
- **モデルチェックポイント**：最良の val_accuracy のモデルのみ保存

---

## 📁 ファイル構成

```
AIsystem/
├── backend/
│   ├── api.py            # FastAPI アプリ（/predict エンドポイント）
│   ├── ai_logic.py        # AI 推論バックエンド（FoodAI クラス）
│   ├── food_model.keras   # 学習済み CNN モデル
│   ├── classes.txt        # 料理クラス一覧（100種）
│   ├── food_calories.csv  # カロリーデータベース
│   ├── requirements.txt   # バックエンドの依存パッケージ
│   └── Dockerfile         # 本番デプロイ用（Render）
├── frontend/               # Next.js フロントエンド（UI）
│   └── src/app/page.tsx    # メイン画面
├── learning_code.txt       # モデル学習コード（Google Colab 用）
└── README.md                # このファイル
```

---

## 🚀 セットアップ & 起動

スマホのカメラ機能（`getUserMedia`）はHTTPS必須のため、開発用の自己署名証明書を使って両サーバーをHTTPSで起動します。

### 0. 開発用証明書の作成（初回のみ）

```bash
mkdir certs && cd certs
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 825 -nodes \
  -subj "/CN=aisystem-dev" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<自分のPCのLAN IP>"
```

`certs/`はコミット対象外（`.gitignore`済み）です。LAN IPが変わったら作り直してください。

### 1. バックエンド（FastAPI）

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --ssl-keyfile ../certs/key.pem --ssl-certfile ../certs/cert.pem
```

`https://localhost:8000` でAPIが起動します（`--host 0.0.0.0`でLAN内の他端末からもアクセス可能）。

### 2. フロントエンド（Next.js）

```bash
cd frontend
npm install
npm run dev
```

`npm run dev`は`--experimental-https`付きで`../certs`の証明書を使うよう設定済みです。ブラウザで `https://localhost:3000` を開きます（自己署名証明書のため警告が出た場合は「詳細設定」から進んでください）。

同じWi-Fi内のスマホからは `https://<PCのLAN IP>:3000` でアクセスできます。バックエンドの接続先はアクセス元ホスト名から自動判定されます（固定したい場合は `frontend/.env.local` の `NEXT_PUBLIC_API_URL` を設定）。

スマホなどLAN内の端末からアクセスする場合は、`frontend/.env.local` に `DEV_LAN_HOST=<自分のPCのLAN IP>` を設定してください（Next.jsの開発サーバーが安全のため未許可のホストからのアクセスをブロックするため）。

---

## ☁️ 本番デプロイ

| レイヤー | サービス | URL |
|----------|----------|-----|
| フロントエンド | [Vercel](https://vercel.com/) | https://mogumogu-mauve.vercel.app |
| バックエンド | [Render](https://render.com/)（Docker, Freeプラン） | https://mogumogu-scanner-api.onrender.com |

- バックエンドは `backend/Dockerfile` を使ってRenderがビルド・起動する
- フロントエンドの環境変数 `NEXT_PUBLIC_API_URL` にバックエンドURLを設定
- バックエンドの環境変数 `ALLOWED_ORIGINS` にフロントエンドURL（`https://`込み）を設定し、CORSを許可
- Renderの無料プランはアクセスがないとスリープするため、しばらく経ってからのアクセスは起動待ちで数十秒かかる場合がある

---

## 🍽️ 対応料理（100種）

<details>
<summary>クリックして全リストを表示</summary>

ごはん / うな重 / ピラフ / 親子丼 / カツ丼 / カレーライス / 寿司 / チキンライス / チャーハン / 天丼 / ビビンバ / トースト / クロワッサン / ロールパン / ぶどうパン / 惣菜パン / ハンバーガー / ピザ / サンドウィッチ / かけうどん / 天ぷらうどん / ざるそば / ラーメン / チャーシューメン / 天津麺 / 焼きそば / スパゲッティ / お好み焼き / たこ焼き / グラタン / 野菜炒め / コロッケ / なすの油味噌 / ほうれん草炒め / 野菜の天ぷら / 味噌汁 / コーンスープ / ウィンナーのソテー / おでん / オムレツ / がんもどきの煮物 / 餃子 / シチュー / 魚の照り焼き / 魚のフライ / 鮭の塩焼 / 鮭のムニエル / 刺身 / さんまの塩焼 / すき焼き / 酢豚 / たたき / 茶碗蒸し / 天ぷら盛り合わせ / 鶏の唐揚げ / 豚カツ / 南蛮漬け / 煮魚 / 肉じゃが / ハンバーグ / ビーフステーキ / 干物 / 豚肉の生姜焼き / 麻婆豆腐 / 焼き鳥 / ロールキャベツ / 卵焼き / 目玉焼き / 納豆 / 冷奴 / 春巻き / 冷やし中華 / チンジャオロース / 角煮 / 筑前煮 / 海鮮丼 / ちらし寿司 / たい焼き / エビチリ / ローストチキン / シュウマイ / オムライス / カツカレー / スパゲッティミートソース / エビフライ / ポテトサラダ / グリーンサラダ / マカロニサラダ / けんちん汁 / 豚汁 / 中華スープ / 牛丼 / きんぴらごぼう / おにぎり / ピザトースト / つけ麺 / ホットドッグ / フライドポテト / 炊き込みご飯 / ゴーヤチャンプル

</details>

---

## ⚙️ 推論の仕様

- 入力画像を **224×224 px** にリサイズして正規化
- Softmax 出力の最大確信度が **10% 未満** の場合は「判定不能」として表示
- 料理名が判定されたあと、`food_calories.csv` から該当するカロリー・目安量を検索して表示
- **TTA（Test-Time Augmentation）** により、1枚の画像を8パターンに変換して推論し、予測確率を平均することで精度を向上
- 結果画面に **AIの予測候補 TOP3** を表示（確信度が低い場合でも候補を確認可能）

---

## 📝 注意事項

- カロリー値はあくまで**目安**です。実際の量・調理法により大きく変わります。
- 登録されていない料理は「判定不能」と表示されます。
- モデルは日本食を中心に学習しているため、洋食・デザートは精度が下がる場合があります。

---

## 👤 開発者・担当

大学の授業（週3コマ×2ヶ月間、4人チーム）で「教師あり学習を用いること」を条件とした開発課題として制作。メンバー全員が日常的にカロリーを気にしていたことから、料理の写真を撮るだけでカロリーが分かるアプリを企画・開発した。

### 担当箇所

- **AIモデルの学習・実装**
  - 推論時にTTA（Test-Time Augmentation）を実装し、1枚の画像を8パターン（反転・明るさ/コントラスト調整・クロップ・回転）に変換して予測を平均することで、モデルの再学習なしに判定精度を向上
  - 確信度10%未満は「判定不能」と表示するなど、実用性を意識した設計
- **リーダー業務**
  - GitHubを用いない開発だったため、結合作業やチームの方針決定・指示出しを担当
  - 開発中、認識の齟齬から異なる仕様のものを作っていたメンバーがいた際は、すり合わせの時間を作りUIを紙に書いて再確認することで認識を合わせた
- **技術選定・アーキテクチャ移行**
  - 当初はPythonのStreamlitでUIを構築していたが、React/Next.jsを学びたいという思いと、背景ぼかしやカードUI・グラデーションボタンなど独自デザインを作り込む中でStreamlitのコンポーネント制約が大きくなったため、フロントエンドをNext.js（React/TypeScript/Tailwind CSS）に、AI推論部分をFastAPIの`/predict`エンドポイントとして切り出す構成に移行
  - Vercel（フロントエンド）・Render（バックエンド、Docker）へのデプロイまで担当
