# Naive Bayes Classifier (nbc)

なんとなく気分で書いた nbc (単純ベイズ識別器)。  
正確には識別ではなくクラスタリングを行うという罠。  

# 使い方

とりあえず [UCI repository][] の [iris][] データをクラスタリングしてみます。
[iris.data][] をダウンロードし、以下のように入力してください。

    ./nbc.rb -f iris.data -y 4

するとなんやかんやでクラスタリングできます。  
デフォルトでは３クラスタに分けるのですが、クラスタ数を自分で死体する場合は、  
以下のように -k option で指定してください。

    ./nbc.rb -f iris.data -y 4 -k 5

上だと５クラスタに分割します。

[UCI repository]: http://archive.ics.uci.edu/ml/
[iris]: http://archive.ics.uci.edu/ml/datasets/Iris
[iris.data]: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

# 中身について

今度暇なときに書きます。