# MLTool: simple modeling tool using ML
単純なデータセットについてモデルを構築し評価するためのツール。
## Quick start
1. python3 actions/action_example.py
2. 実行結果がresults/${idAction}へ出力される。
## Option
本ツールでは、optionを辞書形式で管理する。actions/action_example.pyを参照のこと。
### Detail
- purpose : 何をするか。複数同時に選択可能(例:["searchHyperParameter", "searchParameter", "test"]というふうに入力可能。この場合、ハイパーパラメータサーチ、モデル構築、精度評価が順に行われる)。
    - searchHyperParameter: ハイパーパラメータサーチ
    - searchParameter: 特定のハイパーパラメータを用いてモデル構築
    - test: 構築済モデルで予測＋精度評価
- time2searchHyperParameter: ハイパーパラメータサーチに費やす時間(秒)。
- modelAlgorithm: どのアルゴリズムで訓練するか。
    - DNN
    - RF
- processor: DNNで訓練する際に用いるプロセサ。
    - CPU
    - GPU
- pathDatasetDir: データセットが置かれたディレクトリ。datasets以下を入力(例"/home/s-ogino/MLTool/datasets/egit/isBuggy/4"ならば、"egit/isBuggy/4"を入力する。)
- pathHyperParameter: ハイパーパラメータ設定ファイル(json)へのパス。
- pathParameter: モデルを表すファイル(バイナリ)へのパス
## Dataset
- データセットはcsv形式で管理する。１列目にレコードID、２列めに目的変数、３列目以降に説明変数が並ぶ。
- トレーニング用、バリデーション用、テスト用データセットを、datasets以下のディレクトリにまとめて配置する。ファイル名はそれぞれtrain0.csv, valid0.csv, test.csvとする。datasets/egit/isBuggy/4を参照のこと。
## Result
実行結果はresults/${idAction}へ出力される。
### HyperParameterSearch
- logSearchHyperParameter.txt: optunaによるハイパーパラメータサーチのログ
- hyperparameter.json: 結果として得られた、最善のハイパーパラメータ
### ParameterSearch (=build a model)
- parameter: モデル。
### test
- resultPrediction.csv: 一列目はレコードID, ２列めは正解の値, ３列目はモデルにより予測された値
- report.json: Recall, Precision, f-measure等。
- confusionMatrix.png: 混同行列。
## Author
荻野翔(s-ogino＠ist.osaka-u.ac.jp)
## License
[MIT license](https://en.wikipedia.org/wiki/MIT_License).