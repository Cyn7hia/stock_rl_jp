# 株式取引市場のOpenAI Gym環境とKerasを使った深層強化学習

## 概要

このプロジェクトは、[OpenAI Gym](https://gym.openai.com/)を使用した株式市場取引シミュレーションの一般的な環境を提供します。
トレーニングデータは、Google Financeからダウンロードした毎日の終値ですが、必要に応じて任意のデータを適用できます。
また、[Karpathyの投稿](http://karpathy.github.io/2016/05/31/rl/)の簡単なDeep Q-LearningとPolicy Gradientが含まれています。

実際、このプロジェクトの目的は、株式取引のための最高のRLソリューションを提供するだけでなく、さらなる研究のための一般的なオープン環境を構築することです。
**したがって、モデルのアーキテクチャと機能を操作して、自分のより良いソリューションを手に入れてください。**

## 要件

 - Python2.7以上
 - Numpy
 - HDF5
 - Keras
 - OpenAI Gym

## 使用法

このリポジトリのほとんどのサンプルトレーニングデータは韓国の銘柄です。
目的に合ったトレーニングデータを再ダウンロードする必要があるかもしれません。

上記の要件を満たした後、Deep Q-learningとPolicy Gradientの両方のアルゴリズムのトレーニングを開始できます。

Deep Q-ラーニングで学習する：

    $ python market_dqn.py <リストファイル名> [モデルファイル名]

Policy Gradientで学習する：

    $ python market_pg.py <リストファイル名> [モデルファイル名]

例：

    $ python market_pg.py ./kospi_10.csv pg.h5

このリポジトリの提供されたニューラルネットワークアーキテクチャは学習するには小さすぎます。
あなたがすべての株価データを習得しようとすると、アンダーフィッティングになる可能性があります。
数年間の10〜100銘柄の株式データでちょうどよくフィッティングできました。 （私がチェックしました！！）
したがって、独自のアーキテクチャを再設計する必要があります。
**あなたがより良いものを持っていたら私に教えてください！**

以下は、Policy Gradientを使用したトップ10のKOSPI株データの4年間のトレーニング曲線です。
![トレーニングカーブ](./pg_over_top_10.png)

## TODO
 - オーバーフィッティングをチェックするテスト環境。
 -  PGの学習インターフェースを精緻化する。

## 参照

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)  
[3] [KEras Reinforcement Learning gYM agents, KeRLym](https://github.com/osh/kerlym)  
[4] [Keras plays catch, a single file Reinforcement Learning example](http://edersantana.github.io/articles/keras_rl/)