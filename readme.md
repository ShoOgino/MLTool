# MLTool: simple modeling tool using ML
単純なデータセットについてモデル構築・評価が可能なツール。
## Installation
執筆中
## Quick start
1. actions/action_example.pyを実行する。
2. 実行結果がMLTool/results/${idAction}へ出力される。
## Option
本ツールでは、optionを辞書の形で管理する。actions/action_example.pyを参照のこと。
### Detail
- purpose : 何をするか。複数同時に選択可(例:["searchHyperParameter", "searchParameter", "test"]を入力した場合、ハイパーパラメータサーチ、モデル構築、精度評価が順に行われる)。
    - searchHyperParameter(ハイパーパラメータサーチ)
    - searchParameter(特定のハイパーパラメータを用いてモデル構築)
    - test(構築されたモデルで予測＋精度評価)
- time2searchHyperParameter : ハイパーパラメータサーチに費やす時間(秒)。
- modelAlgorithm            : どのアルゴリズムで学習させるか。
    - DNN
    - RF
- processor: DNNを用いる場合に用いるプロセサ。
    - CPU
    - GPU
- pathDatasetDir: データセット(train0.csv, valid.csv, test.csv)が置かれたディレクトリ。datasets以下を入力(例"/home/s-ogino/MLTool/datasets/egit/isBuggy/4"ならば、"egit/isBuggy/4"を入力する。)
- pathHyperParameter : ハイパーパラメータ設定ファイルへのパス。
- pathParameter : モデルを表すファイルへのパス
### Dataset
- データセットはレコードの集合であり、レコードは{レコードID(1個), 目的変数(1個), 説明変数(n個)}というデータ組と定義する。
- データセットはcsv形式で管理する。１列目にレコードID、２列めに目的変数、３列目以降に説明変数が並びます。MLTool/datasets/egit/isBuggy/4/train0.csvを参照のこと。
- トレーニング用、バリデーション用、テスト用データセットを、MLTool/datasets/より下に配置する。ファイル名はそれぞれtrain0.csv, valid0.csv, test.csvとする。
### Result
実行結果として得られた「ハイパーパラメータ」「モデル」「予測結果」等はMLTool/results/${idAction}/へ出力されます。
## Author
- 荻野翔(s-ogino＠ist.osaka-u.ac.jp)
## License
[MIT license](https://en.wikipedia.org/wiki/MIT_License).