# Mindfulness BGM Generator

[English](README.md) | 日本語

瞑想やマインドフルネスのために、持続音、和音、波の環境音、打楽器をリアルタイムで生成するアプリです。GUIは初回起動時に英語で表示され、日本語に切り替えられます。技術仕様と変更履歴は [英語版README](README.md) を参照してください。

## インストールと起動

Python 3.10以上と音声出力機器が必要です。

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
python -m venv .venv
```

Windows PowerShellの場合:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

macOS / Linuxの場合:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Ubuntu / Debianでは、先に `sudo apt-get install portaudio19-dev python3-all-dev` を実行してください。macOSでは `brew install portaudio` でPortAudioを導入します。WindowsではsounddeviceにPortAudioが同梱されており、PyAudioやpipwinは不要です。

## 表示言語

画面上部の **Language** で **日本語** を選ぶと、ボタン、設定項目、プリセットの説明、再生状態、アプリのエラーメッセージが日本語になります。**English** を選ぶと英語に戻ります。

選択した言語は次回の起動にも引き継がれます。再生中も切り替えられ、音の設定やセッションは維持されます。出力機器の名前と技術的なエラー詳細は、音声システムが返した表記のまま表示されます。

## 基本の使い方

1. **過ごしかた（Preset）** からプリセットを選びます。
2. **座る時間（Session length）** を15・20・30・60分、または **時間を決めない（Untimed）** に設定します。初期値は20分です。
3. **音の大きさ（Volume）** と **出力先（Output device）** を調整します。機器を接続し直した場合は **再検出（Refresh）** を押します。
4. **はじめる（Begin）** で再生します。再生中は設定欄が隠れ、静坐用の画面に変わります。
5. 途中で終える場合は **終える（End）** を押します。音は徐々に消えます。

| 英語のプリセット名 | 日本語表示 | 内容 |
| --- | --- | --- |
| Incense & Stillness | 線香と静坐 | 静かな低い持続音。打楽器、波の環境音、左右の周波数差は無効です。 |
| Sitting with Resonance | 余韻に座る | 持続音に、90〜150秒間隔のクリスタルボウルを加えます。 |
| Distant Shore | 遠い水辺 | 持続音の奥に、ごく薄く波の環境音を添えます。 |
| Shō Resonance | 笙のひびき | 笙をイメージした和音が呼吸のように膨らみ、一音ずつ移ろいます。 |
| Custom | 自分で調整 | 楽器、和音、環境音、間隔を好みに合わせて設定します。 |

再生開始時は8秒かけて音がなじみます。時間を指定したセッションでは最後の20秒でフェードアウトします。瞑想プリセットはおりんで始まり、時間指定時は終わりにもおりんが鳴り、その余韻が消えてから停止します。タイマーは線香の燃焼時間とは独立しています。

## 音を細かく調整する

**音を細かく調整する（Advanced sound settings）** を開くと、次の項目を設定できます。音の設定を変更すると、プリセットは **自分で調整** に切り替わり、通常の生成モードになります。設定の変更は再生を停止してから行います。

- **楽器**: ランダム、チベタンベル、スリットドラム、ハンドパン、クリスタルボウル、打楽器なし。
- **海の環境音**: 波の音を0〜100%で調整します。
- **和音と音色**: 和音の変化、倍音の豊かさ、和音の種類、進行スタイルを選択します。
- **30分かけて和音の変化を調整する**: セッションの進行に応じて和音の動きを変化させます。
- **左右の音にわずかな周波数差をつける**: ステレオの周波数差を有効にします。
- **楽器の間隔**: 楽器ごとの最短・最長間隔を秒単位で指定します。最短を0秒にすると、その楽器を無効にします。

## コマンドラインで使う

コマンドライン版のヘルプやオプションは英語です。

```bash
python main.py --help
python main.py --instrument handpan --handpan 10-20
python main.py --modulation-mode stable --harmonic-richness minimal
```

Raspberry Pi Zero 2W向けの軽量版:

```bash
python main_rpiz2w.py --help
python main_rpiz2w.py
```

すべてのオプションは [英語版の利用方法](README.md#usage) を参照してください。拡張和音設定は `main.py` で利用できます。

## 困ったとき

- **音が出ない**: OSの音量と出力先を確認し、GUIで出力先を再検出してください。通常版は48 kHzのステレオ出力を使用します。
- **再生エラー**: 音声機器の接続と出力先を確認してください。エラーや再生中断が起きると、GUIは停止状態に戻ります。
- **起動できない**: `app.py` を実行するPython環境に `requirements.txt` の依存関係をインストールしてください。
- **音が途切れる**: 他の音声アプリを閉じ、CPU負荷を確認してください。

## 動作確認

実際の音声機器を開かずに回帰テストを実行できます。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS / Linuxでは、仮想環境を有効にして `python -m unittest discover -s tests -v` を実行します。聴感とRaspberry Pi実機での性能は、実際の機器での確認が必要です。
