<p align="center">
  <img src="assets/logo-on-dark.jpg" alt="ALICE Streaming Protocol" width="600">
</p>

# ALICE Streaming Protocol (libasp)

Rustで書かれた高性能ビデオストリーミングコーデック（Pythonバインディング付き）

**A.L.I.C.E.** = Adaptive Low-bandwidth Image Codec Engine

## 特徴

- **超低帯域幅**: 従来のコーデックと比較して100-1000倍の削減
- **手続き的生成**: ピクセルではなく数学的記述を送信
- **クロス言語対応**: FlatBuffersシリアライズ（C++, Go, Java, Python, TypeScript等）
- **ゼロコピー**: デシリアライズなしでバッファに直接アクセス
- **並列処理**: Rayonによるマルチコア性能の活用
- **ゼロコピーPythonバインディング**: PyO3経由でNumPy配列に直接アクセス
- **ベアメタル性能**: コンパイル時CRC32テーブル、GIL解放、バッファ再利用

## パフォーマンスハイライト

| 最適化 | 改善効果 |
|-------|---------|
| CRC32ルックアップテーブル（コンパイル時生成） | 2.8-5.3倍高速 |
| バッファ再利用（ストリーミング） | 77倍高速 |
| 重い計算時のGIL解放 | Python完全並列化 |
| 直接PyArray割り当て | 中間コピーゼロ |

## パケットタイプ

| タイプ | 名前 | 説明 | 典型的なサイズ |
|-------|------|------|--------------|
| I | キーフレーム | 完全な手続き的記述 | 10-100KB (8K) |
| D | デルタ | 差分更新 + モーションベクトル | 1-10KB |
| C | 補正 | ROIベースのピクセル補正 | 可変 |
| S | 同期 | フロー制御コマンド | < 100 bytes |

## クロス言語サポート

ASPはFlatBuffersをシリアライズに使用し、どの言語からでもゼロコピーアクセスを可能にします。

**各言語向けコード生成:**

```bash
# FlatBuffersコンパイラのインストール
brew install flatbuffers  # macOS
apt install flatbuffers-compiler  # Ubuntu

# コード生成
flatc --cpp -o generated/ schemas/asp.fbs     # C++
flatc --go -o generated/ schemas/asp.fbs      # Go
flatc --java -o generated/ schemas/asp.fbs    # Java
flatc --ts -o generated/ schemas/asp.fbs      # TypeScript
flatc --python -o generated/ schemas/asp.fbs  # Python
```

**C++での使用例:**

```cpp
#include "asp_generated.h"

// D-Packetの読み取り（ゼロコピー！）
auto packet = Alice::Asp::GetAspPacketPayload(buffer);
auto d_packet = packet->payload_as_DPacketPayload();
auto mvs = d_packet->motion_vectors();

for (auto mv : *mvs) {
    printf("ブロック (%d, %d): dx=%d, dy=%d\n",
           mv->block_x(), mv->block_y(), mv->dx(), mv->dy());
}
```

## インストール

### ソースから（Rust）

```bash
git clone https://github.com/ext-sakamoro/ALICE-Streaming-Protocol.git
cd ALICE-Streaming-Protocol
cargo build --release
```

### Pythonパッケージ

```bash
cd ALICE-Streaming-Protocol
pip install maturin
maturin develop --release
```

注意: Python 3.14以降では、ビルド前に `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` を設定してください。

## 使用方法

### Rust

```rust
use libasp::{AspPacket, IPacketPayload, DPacketPayload, MotionVector, codec};

// キーフレームを作成
let payload = IPacketPayload::new(1920, 1080, 30.0);
let packet = AspPacket::create_i_packet(1, payload)?;

// 再利用可能なバッファにシリアライズ（ホットループでゼロアロケーション）
let mut buffer = Vec::with_capacity(65536);
packet.write_to_buffer(&mut buffer)?;

// または固定スタックバッファを使用（真のゼロアロケーション）
let mut stack_buffer = [0u8; 65536];
let written = packet.serialize_into(&mut stack_buffer)?;

// 並列Diamond Searchによるモーション推定
let mvs = codec::estimate_motion_parallel(
    &current_frame,
    &previous_frame,
    1920, 1080,
    16, 16,  // block_size, search_range
    codec::SearchAlgorithm::DiamondSearch,
    256,     // 早期終了閾値
);

// モーションベクトルでデルタパケットを作成
let mut d_payload = DPacketPayload::new(1);
for mv in mvs {
    d_payload.add_motion_vector(mv);
}
let d_packet = AspPacket::create_d_packet(2, d_payload)?;
```

### Python（NumPyゼロコピー）

```python
import numpy as np
import libasp

# フレームをNumPy配列として読み込み (H, W) uint8
current = np.array(current_frame, dtype=np.uint8)
previous = np.array(previous_frame, dtype=np.uint8)

# NumPyゼロコピーI/Oによるモーション推定
# (N, 5) int32配列を返す: [block_x, block_y, dx, dy, sad]
mvs = libasp.estimate_motion_numpy(current, previous, block_size=16, search_range=16)

# または繰り返し呼び出し用にMotionEstimatorクラスを使用
estimator = libasp.MotionEstimator(block_size=16, search_range=16, algorithm="diamond")
mvs = estimator.estimate_numpy(current, previous)

# 色抽出（k-means、(N, 3) uint8配列を返す）
colors = libasp.extract_colors(pixels, num_colors=5)

# 重み付き
colors, weights = libasp.extract_colors_with_weights(pixels, num_colors=5)

# DCT変換
dct = libasp.DctTransform(block_size=8, quality=75)
coefficients = dct.forward(block)
reconstructed = dct.inverse(coefficients)

# ROI検出（(N, 6) float32配列を返す: [x, y, w, h, type, confidence]）
regions = libasp.detect_roi(frame, previous_frame, width, height)

# パケット作成
i_packet = libasp.create_i_packet(sequence=1, width=1920, height=1080, fps=30.0)
d_packet = libasp.create_d_packet_numpy(sequence=2, ref_sequence=1, motion_vectors=mvs)

# パケット解析
packet_type, sequence, payload_size = libasp.parse_packet(packet_bytes)
```

## ベンチマーク

ベンチマークの実行:

```bash
cargo bench --no-default-features
```

### モーション推定（Apple Silicon）

| 解像度 | アルゴリズム | 時間 |
|-------|------------|------|
| 256x256 | Diamond Search | 19.6 µs |
| 512x512 | Diamond Search | 24.5 µs |
| 1024x1024 | Diamond Search | 42.0 µs |
| **1080p** | Diamond Search | **64.5 µs** |

### CRC32パフォーマンス

コンパイル時生成ルックアップテーブルにより、テーブルなし実装と比較して2.8-5.3倍の高速化を実現。

### テストスイート

```bash
cargo test --no-default-features
# running 67 tests
# test result: ok. 67 passed; 0 failed
```

## アーキテクチャ

```
libasp/
├── src/
│   ├── lib.rs          # ライブラリルート、バージョン情報
│   ├── types.rs        # コア型（MotionVector, Color, Rect等）
│   ├── header.rs       # パケットヘッダー（16バイト）+ CRC32
│   ├── packet.rs       # パケットペイロード（I/D/C/S）+ シリアライズ
│   ├── codec/
│   │   ├── mod.rs      # コーデックモジュールエクスポート
│   │   ├── motion.rs   # モーション推定（Diamond/Hexagon/TSS/Full）
│   │   ├── dct.rs      # DCT/IDCT変換 + 量子化
│   │   ├── color.rs    # 色抽出（k-means, median cut）
│   │   └── roi.rs      # ROI検出（エッジ、動き、顔）
│   └── python.rs       # PyO3 + NumPyバインディング
├── benches/
│   └── motion_estimation.rs  # Criterionベンチマーク
└── Cargo.toml
```

## 主要データ構造

### MotionVector（16バイト、キャッシュ最適化）

```rust
#[repr(C)]
pub struct MotionVector {
    pub block_x: u16,    // ブロックX位置（4K+対応）
    pub block_y: u16,    // ブロックY位置
    pub dx: i16,         // 水平方向変位
    pub dy: i16,         // 垂直方向変位
    pub sad: u32,        // 絶対差分和
    pub _reserved: u32,  // 将来使用 / アラインメント
}
```

### MotionVectorCompact（2バイト、転送用）

```rust
#[repr(C, packed)]
pub struct MotionVectorCompact {
    pub dx: i8,  // -128から127ピクセル
    pub dy: i8,
}
```

## フィーチャーフラグ

```toml
[features]
default = ["python"]
python = ["pyo3", "numpy"]   # Pythonバインディング
wasm = ["wasm-bindgen"]      # WebAssemblyサポート
simd = []                    # 明示的SIMD（自動検出）
```

## ライセンス

**デュアルライセンスモデル** - 詳細は[LICENSE](LICENSE)を参照。

| コンポーネント | ライセンス | 目的 |
|--------------|-----------|------|
| プロトコル仕様 | MIT | グローバル標準採用 |
| レシーバー（デコーダー） | MIT | あらゆるデバイスで動作 |
| センダー（エンコーダコア） | MIT | 超低帯域幅技術の民主化 |
| センダー（エンタープライズ） | 商用 | 大規模インフラ向け |

**帰属表示**: 無料エンコーダ使用者はストリームメタデータに「Streamed via ALICE Protocol (ASP)」を含める必要があります。

## 関連プロジェクト

| プロジェクト | 説明 |
|------------|------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | 手続き的生成圧縮エンジン。生データではなく「データの生成方法」を保存し、数学的パターンに対して1400倍以上の圧縮を実現。 |

両プロジェクトは共通の哲学を共有しています：**データそのものではなく、生成プロセスをエンコードする**。

## 作者

坂本 師哉 (Moroya Sakamoto)

---

[English README](README.md)
