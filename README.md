# Mindfulness BGM Generator

A dynamic ambient music generator designed for meditation and mindfulness practices. Creates continuous, evolving soundscapes with harmonious chords, natural ambient sounds, and traditional meditation instruments.

## Features

- **Dynamic Sound Generation**: Continuously evolving harmonies and textures
- **Multiple Sound Types**:
  - Harmonic-rich tones
  - Pure tones
  - Soft pad sounds
  - Warm tones
  - Bell-like sounds
- **Meditation Instruments** (one instrument selected per session):
  - Tibetan bells (Tingsha)
  - Slit drums
  - Handpan sounds
- **Natural Ambient Sounds**: Ocean waves and wind with enhanced volume
- **Breathing Rhythm Synchronization**: Subtle volume modulation synced to typical breathing patterns
- **Real-time Audio Processing**: Low-latency sound generation
- **Customizable Parameters**: Adjust instrument intervals and ambient mix ratio
- **Smooth Transitions**: Crossfading between chords and sound types
- **Simple Reverb Effect**: Adds spatial depth to the soundscape
- **Mindfulness-oriented Harmonies**: Uses perfect fifths, fourths, and open intervals

## Requirements

- Python 3.7+
- NumPy
- SoundDevice
- pyaudio (optional, for some systems)

For the Streamlit web app:

- Streamlit

## Installation

1. Clone this repository:

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Version (main.py)

Run with default settings (randomly selects one instrument):

```bash
python main.py
```

### Web App Version (app.py)

Run the Streamlit application:

```bash
streamlit run app.py
```

The web app provides a graphical interface with:

- Start/Stop controls
- Instrument selection
- Ambient sound mix adjustment
- Interval settings for each instrument

### Command-line Options

- `--instrument`: Choose specific instrument ('bell', 'drum', or 'handpan')
- `--bell`: Tibetan bell interval in seconds (e.g., '15-45' or '30' or '0' to disable)
- `--drum`: Slit drum interval in seconds (e.g., '8-25' or '15' or '0' to disable)  
- `--handpan`: Handpan interval in seconds (e.g., '12-30' or '20' or '0' to disable)
- `--ambient`: Ambient sound ratio (0-1, default: 0.3, '0' to disable)

### Examples

Use only Tibetan bells:

```bash
python main.py --instrument bell
```

Use only slit drum:

```bash
python main.py --instrument drum
```

Use only handpan:

```bash
python main.py --instrument handpan
```

Set custom intervals for the selected instrument:

```bash
python main.py --instrument drum --drum 10-20
```

Adjust ambient sound mix:

```bash
python main.py --ambient 0.5
```

Ambient sounds only (no percussion):

```bash
python main.py --bell 0 --drum 0 --handpan 0
```

## New Features

### Single Instrument per Session

Each time you run the program, it automatically selects one of the three meditation instruments (Tibetan bell, slit drum, or handpan) to use throughout the session. This creates a more focused meditative experience.

### Enhanced Musical Harmony

- **Mindfulness-oriented chord progressions**: Uses open intervals, perfect fifths, fourths, and single notes instead of traditional Western harmonies
- **Harmonic percussion**: Instruments play notes that harmonize with the current background chord
- **Natural rhythm patterns**: Instruments play 1-3 notes with musical spacing (quarter note to whole note intervals)

### Improved Ambient Sounds

- Enhanced volume and presence of natural sounds
- Richer ocean wave patterns with multiple frequencies
- More dynamic wind sounds
- Added low-frequency rumble for depth

## Technical Details

### Audio Processing

- **Sample Rate**: 44100 Hz
- **Buffer Size**: 1024 frames
- **Channels**: Stereo
- **Output Format**: Float32

### Musical Elements

- **Scale**: A Pentatonic (A, C, E, A, C)
- **Chord Types** (Mindfulness-oriented):
  - Single notes (drone)
  - Octaves
  - Perfect fifths
  - Perfect fourths
  - Quartal harmonies
  - Sus2 chords
  - Open voicings

### Sound Generation

- Pure sine wave synthesis
- Harmonic addition for timbral richness
- Exponential envelope generators
- Phase modulation for stereo width
- Soft limiting to prevent clipping

## Architecture

The application uses a multi-threaded architecture:

- Main thread: Audio callback and signal processing
- Event scheduler thread: Manages chord and sound type changes
- Single instrument scheduler thread: Controls the selected percussion timing

## Customization

You can modify the following constants in the code:

- `VOLUME`: Overall output volume
- `MIN_EVENT_INTERVAL`: Minimum time between sound changes
- `MAX_EVENT_INTERVAL`: Maximum time between sound changes
- `FADE_TIME`: Crossfade duration
- `BREATH_CYCLE`: Breathing rhythm frequency
- `BASE_FREQS`: Musical scale frequencies
- `DEFAULT_AMBIENT_RATIO`: Default ambient sound mix level (0.3)

## Known Issues

- On some systems, you may need to specify the audio device explicitly
- Very low buffer sizes may cause audio glitches on slower systems

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[MIT License](LICENSE)

## Acknowledgments

Inspired by traditional meditation music and modern sound therapy techniques. Special thanks to the Python audio processing community.

---

# マインドフルネスBGMジェネレーター

瞑想とマインドフルネス実践のために設計されたダイナミックなアンビエント音楽ジェネレーターです。調和のとれたコード、自然なアンビエントサウンド、伝統的な瞑想楽器を使用して、継続的に進化するサウンドスケープを作成します。

## 特徴

- **動的なサウンド生成**: 継続的に進化するハーモニーとテクスチャー
- **複数のサウンドタイプ**:
  - ハーモニックリッチなトーン
  - ピュアトーン
  - ソフトパッドサウンド
  - 温かみのあるトーン
  - ベルのようなサウンド
- **瞑想楽器** (セッションごとに1つの楽器を選択):
  - チベタンベル（ティンシャ）
  - スリットドラム
  - ハンドパンサウンド
- **自然なアンビエントサウンド**: 強化されたボリュームの海の波と風
- **呼吸リズムの同期**: 一般的な呼吸パターンに同期した微妙なボリューム変調
- **リアルタイムオーディオ処理**: 低レイテンシのサウンド生成
- **カスタマイズ可能なパラメータ**: 楽器の間隔とアンビエントミックス比率の調整
- **スムーズなトランジション**: コードとサウンドタイプ間のクロスフェード
- **シンプルなリバーブエフェクト**: サウンドスケープに空間的な深さを追加
- **マインドフルネス指向のハーモニー**: 完全5度、4度、開放的な音程を使用

## 必要な環境

- Python 3.7以上
- NumPy
- SoundDevice
- pyaudio（一部のシステムでオプション）

Streamlitウェブアプリの場合：

- Streamlit

## インストール

1. このリポジトリをクローン：

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
```

2. 必要なパッケージをインストール：

```bash
pip install -r requirements.txt
```

## 使用方法

### コマンドライン版 (main.py)

デフォルト設定で実行（楽器をランダムに選択）：

```bash
python main.py
```

### ウェブアプリ版 (app.py)

Streamlitアプリケーションを実行：

```bash
streamlit run app.py
```

ウェブアプリは以下の機能を持つグラフィカルインターフェースを提供します：

- 開始/停止コントロール
- 楽器の選択
- アンビエントサウンドミックスの調整
- 各楽器の間隔設定

### コマンドラインオプション

- `--instrument`: 特定の楽器を選択（'bell'、'drum'、または 'handpan'）
- `--bell`: チベタンベルの間隔（秒）（例：'15-45' または '30'、'0' で無効化）
- `--drum`: スリットドラムの間隔（秒）（例：'8-25' または '15'、'0' で無効化）  
- `--handpan`: ハンドパンの間隔（秒）（例：'12-30' または '20'、'0' で無効化）
- `--ambient`: アンビエントサウンド比率（0-1、デフォルト：0.3、'0' で無効化）

### 使用例

チベタンベルのみを使用：

```bash
python main.py --instrument bell
```

スリットドラムのみを使用：

```bash
python main.py --instrument drum
```

ハンドパンのみを使用：

```bash
python main.py --instrument handpan
```

選択した楽器にカスタム間隔を設定：

```bash
python main.py --instrument drum --drum 10-20
```

アンビエントサウンドミックスを調整：

```bash
python main.py --ambient 0.5
```

アンビエントサウンドのみ（パーカッションなし）：

```bash
python main.py --bell 0 --drum 0 --handpan 0
```

## 新機能

### セッションごとの単一楽器

プログラムを実行するたびに、3つの瞑想楽器（チベタンベル、スリットドラム、ハンドパン）のうち1つを自動的に選択し、セッション全体で使用します。これにより、より集中した瞑想体験が生まれます。

### 強化された音楽的ハーモニー

- **マインドフルネス指向のコード進行**: 伝統的な西洋のハーモニーの代わりに、開放的な音程、完全5度、4度、単音を使用
- **ハーモニックパーカッション**: 楽器が現在の背景コードと調和する音を演奏
- **自然なリズムパターン**: 楽器が音楽的な間隔（4分音符から全音符の間隔）で1-3音を演奏

### 改善されたアンビエントサウンド

- 自然音の強化されたボリュームと存在感
- 複数の周波数を持つより豊かな海の波のパターン
- よりダイナミックな風の音
- 深さのための低周波ランブルの追加

## 技術詳細

### オーディオ処理

- **サンプルレート**: 44100 Hz
- **バッファサイズ**: 1024フレーム
- **チャンネル**: ステレオ
- **出力フォーマット**: Float32

### 音楽要素

- **スケール**: Aペンタトニック（A、C、E、A、C）
- **コードタイプ**（マインドフルネス指向）:
  - 単音（ドローン）
  - オクターブ
  - 完全5度
  - 完全4度
  - 4度重ねハーモニー
  - Sus2コード
  - 開放的なボイシング

### サウンド生成

- 純粋な正弦波合成
- 音色の豊かさのためのハーモニック加算
- 指数エンベロープジェネレーター
- ステレオ幅のための位相変調
- クリッピング防止のためのソフトリミッティング

## アーキテクチャ

アプリケーションはマルチスレッドアーキテクチャを使用：

- メインスレッド: オーディオコールバックと信号処理
- イベントスケジューラスレッド: コードとサウンドタイプの変更を管理
- 単一楽器スケジューラスレッド: 選択されたパーカッションのタイミングを制御

## カスタマイズ

コード内の以下の定数を変更できます：

- `VOLUME`: 全体の出力ボリューム
- `MIN_EVENT_INTERVAL`: サウンド変更間の最小時間
- `MAX_EVENT_INTERVAL`: サウンド変更間の最大時間
- `FADE_TIME`: クロスフェード時間
- `BREATH_CYCLE`: 呼吸リズム周波数
- `BASE_FREQS`: 音階の周波数
- `DEFAULT_AMBIENT_RATIO`: デフォルトのアンビエントサウンドミックスレベル（0.3）

## 既知の問題

- 一部のシステムでは、オーディオデバイスを明示的に指定する必要がある場合があります
- 非常に低いバッファサイズは、遅いシステムでオーディオのグリッチを引き起こす可能性があります

## コントリビューション

コントリビューションは歓迎です！プルリクエストを送信したり、バグや機能リクエストのためのイシューを作成したりしてください。

## ライセンス

[MITライセンス](LICENSE)

## 謝辞

伝統的な瞑想音楽と現代のサウンドセラピー技術に触発されています。Pythonオーディオ処理コミュニティに特別な感謝を。
