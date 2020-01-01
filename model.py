import torch
import torch.nn as nn


'''
########################################################################
#########  PytorchによるEntity Embeddingを利用した回帰用DNNの実装  #########
########################################################################

factorize(※)されたカテゴリカル変数(前半)と数値変数(後半)のインプットを持つ回帰用DNN。
カテゴリカル変数それぞれに固有のEmbedding Layerを用意し、対応する密ベクトル空間へのマッピングを行う(Entity Embedding)。
(例: 月(0),火(1),水(2),木(3),金(4),土(5),日(6) の要素を持つ辞書数7のカテゴリカル変数をEmbedding Layerにより2次元空間にマッピング)
Embeddingされた各カテゴリカル特徴量と数値変数を結合して全結合層に入力し、回帰を行う。

(※)factorize: カテゴリカル変数が持つ各要素を、対応する正のインデックスに変換する処理。


#############################
#####  モデル作成時の入力  #####
#############################

    categorical_dicts_to_dims   :   複数の[辞書数, Embedding後の次元数]の対で指定される、
                                    各カテゴリカル変数それぞれに対するEmbedding Layerの設計。
                                    空のリストを指定すると、カテゴリカル変数なしの通常の全結合ネットワークになる。
                                        例: [[7, 2], [10, 3]]   (2つのカテゴリカル変数を持ち、それぞれ辞書数が7と10の場合)
       num_numerical_features   :   数値変数の数
       fc_layers_construction   :   各レイヤーのノード数を格納したリストで指定される、全結合層の設計。
                                        例: [10, 10, 5]
          dropout_probability   :   (オプション) FCレイヤーのドロップアウト率。 (default:0.)


############################
#####  forward時の入力  #####
############################

                        input   :   前半にfactorizeされたカテゴリカル変数、後半に数値変数を持つ入力データのミニバッチ。
                                    カテゴリカル変数はモデル作成時の設計(変数の数、変数それぞれの辞書数)に従う必要がある。
'''


class CategoricalDnn(nn.Module):
    def __init__(self,
                 categorical_dicts_to_dims,
                 num_numerical_features,
                 fc_layers_construction,
                 dropout_probability=0.):
        super(CategoricalDnn, self).__init__()

        self._make_embedding_layers(categorical_dicts_to_dims)

        self._make_fc_layers(num_numerical_features, fc_layers_construction, dropout_probability)

    def _make_embedding_layers(self, categorical_dicts_to_dims):
        """
        カテゴリカル変数それぞれのEmbedding Layerを定義。
        プロパティ：
            self.embedding_layer_list         :   各カテゴリカル変数それぞれに適用するEmbedding Layerのリスト
            self.num_categorical_features     :   Embedding前のカテゴリカル変数の総数
            self.num_embedded_features        :   Embedding後のカテゴリカル特徴量の総数
            self.num_each_embedded_features   :   Embedding前の各カテゴリカル変数それぞれに対するEmbedding後の特徴量の数
        """
        self.num_categorical_features = len(categorical_dicts_to_dims)
        self.num_embedded_features = 0
        self.num_each_embedded_features = []
        self.embedding_layer_list = nn.ModuleList()
        for dict_to_dim in categorical_dicts_to_dims:
            num_dict = dict_to_dim[0]
            target_dim = dict_to_dim[1]
            self.embedding_layer_list.append(
                nn.Sequential(
                    nn.Embedding(num_dict, target_dim),
                    nn.BatchNorm1d(target_dim)
                )
            )
            self.num_embedded_features += target_dim
            self.num_each_embedded_features.append(target_dim)

    def _make_fc_layers(self, num_numerical_features, fc_layers_construction, dropout_p):
        """
        全結合層の入力層・隠れ層・出力層を定義。
        プロパティ：
            self.fc_layer_list   :   Embeddingされたカテゴリカル特徴量と数値特徴量を入力とするFCレイヤーのリスト
            self.output_layer    :   出力層
        """
        num_input = self.num_embedded_features + num_numerical_features
        self.fc_layer_list = nn.ModuleList()
        for num_output in fc_layers_construction:
            self.fc_layer_list.append(
                nn.Sequential(
                    nn.Dropout(dropout_p) if dropout_p else nn.Sequential(),
                    nn.Linear(num_input, num_output),
                    nn.BatchNorm1d(num_output),
                    nn.ReLU(inplace=True)
                )
            )
            num_input = num_output
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_p) if dropout_p else nn.Sequential(),
            nn.Linear(num_input, 1)
        )

    def forward(self, input):
        # カテゴリカル変数と数値変数の分離
        categorical_input = input[:, 0:self.num_categorical_features].long()
        numerical_input = input[:, self.num_categorical_features:]

        # カテゴリカル変数のEmbedding
        embedded = torch.zeros(input.size()[0], self.num_embedded_features)
        start_index = 0
        for i, emb_layer in enumerate(self.embedding_layer_list):
            gorl_index = start_index + self.num_each_embedded_features[i]
            embedded[:, start_index:gorl_index] = emb_layer(categorical_input[:, i])
            start_index = gorl_index

        # Embeddingされた特徴量と数値特徴量を結合して全結合層へ入力
        out = torch.cat([embedded, numerical_input], axis=1)
        for hidden_layer in self.fc_layer_list:
            out = hidden_layer(out)
        return self.output_layer(out)



#### test
if __name__ == "__main__":

    # 前半4つがカテゴリカル変数、後半3つが数値変数
    # batch sizeは2
    test_input = torch.tensor([[0., 3., 7., 4., 3.2, 1.22, -8.3],
                               [2., 1., 6., 3., 1.3, 0.56, -1.67]])

    # カテゴリカル変数それぞれについて、辞書数とembedding後の次元数を定義
    embedding_construction = [[3, 2], [4, 2], [10, 4], [7, 3]]

    # modelの定義
    model = CategoricalDnn(categorical_dicts_to_dims=embedding_construction,
                           num_numerical_features=3,
                           fc_layers_construction=[30, 20, 20],
                           dropout_probability=0.)
    model.train()
    test_out = model(test_input)

    # カテゴリカル変数なしのモデル(通常の全結合ネットワーク)も作れる
    model2 = CategoricalDnn(categorical_dicts_to_dims=[],
                            num_numerical_features=7,
                            fc_layers_construction=[10, 10, 5, 5],
                            dropout_probability=0.2)
    model2.train()
    test_out2 = model2(test_input)





