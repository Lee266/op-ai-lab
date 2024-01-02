# op-ai-lab

## 概要

op-ai-labはAIに関する研究と開発を行うプロジェクトフォルダです。
[op-ai-labのレポジトリ](https://github.com/Lee266/op-ai-lab)
当プロジェクトは、ディープラーニング、画像処理のAI分野に焦点を当てています。

## 必須環境

- cuda>=12.1
- nvidia driver

## 準備

### Slack apiの利用

通知にSlack apiを利用しているので、アカウントを作成してください。
[Slackのサイト](https://api.slack.com/)
作成後、Create a new appからプロジェクト用のアプリを作成し、Feature/Incoming WebhooksからWebhookURLをCOPYしてenvファイルに追加してください。

### .envファイルの修正

.envファイルがない場合は.example.envからコピーしてください。  
.envファイル内の{}の値を自分が設定したい値に変更してください。

## 始めるには

### 初めに

始めるには、次のURLからプロジェクトレポジトリからクローンしプロジェクトレポジトリのREADMEに従ってください.  
[プロジェクトリポジトリ](https://github.com/Lee266/op-ai-monorepo)

上記が成功した場合、以下を開いてください。
jupyterlabを使いたい場合は: <http://localhost:8888>
flaskを使いたい場合は: <http://localhost:5000>

## 使用されている主なパッケージ

- python v3.10
- torch v2.1.2
- torchvision v0.16.0
- timm v0.6.12
- flask

## フォルダ構成

`src` フォルダを使用しています。

- app: flaskで使うroute(api)の設定など
- datasets: データセットの作成するフォルダ
- dev: 主に実験をするフォルダ。学習、推論、可視化など
- models: timmのモデルの登録や、モデルの作成など
- util: 共通に使われる関数

## その他

### テスト済み環境

| GPU                 | Driver Version | CUDA Version | OS        |
|---------------------|-----------------|--------------|-----------|
| NVIDIA RTX 3070     | 536.40          | 12.2         | Windows 11 |
| NVIDIA RTX 4090     | 546.33          | 12.3         | Windows 11 |
