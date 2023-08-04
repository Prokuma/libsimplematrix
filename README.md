# libsimplematrix
腕試し的にテキトーに作った行列計算ライブラリです。
いくつかの線形代数演算及び全結合層のニューラルネットワーク順伝播に対応しています。
ドキュメントは気が向いたら書きますがヘッダーファイル読んだら使い方はわかると思います。

## ビルド方法
### 静的ライブラリ
```bash
make static
```
###　テストプログラム
```bash
make test
```
### コンパイルオプション（共通）
OpenMP
```
make (some) USE_MP=1
```