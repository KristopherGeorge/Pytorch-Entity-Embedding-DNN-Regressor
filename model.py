import torch
import torch.nn as nn


'''
########################################################################################
######### Implementation of DNN-regressor using Entity Embedding[1] by Pytorch #########
########################################################################################

An implementation of DNN for regression with input of factorized (*) categorical variables (first half) and numerical variables (second half).
Preparing a unique Embedding Layer for each categorical variable and perform mapping to the corresponding dense vector space (Entity Embedding).
(Example: Mapping a categorical variable with 7 categories which elements are Mon.(0), Tue.(1), Wed.(2), Thu.(3), Fri.(4), Sat.(5), Sun.(6) to 2D space by an Embedding Layer)
Regression is performed by combining the embedded categorical features and numerical variables and inputting them to Fully Connected Layers.

(*) Factorization: The process of converting each element of a categorical variable into a corresponding positive index.


#######################################
##### Input when creating a model #####
#######################################

 categorical_nuniques_to_dims  : Embedding Layer design for each categorical variable, specified by multiple pairs of
                                 [(number of dictionaries), (dimension of the embedded feature)].
                                 Specifying an empty list results in
                                 a regular fully connected network without categorical variables.
                                      Example: [[7, 2], [10, 3]]
                                               (with two categorical variables
                                                which have 7 and 10 dictionaries, respectively)
       num_numerical_features  : Number of numerical variables
       fc_layers_construction  : Design of the Fully Connected Layers, specified by
                                 a list containing the number of nodes in each layer.
                                      Example: [10, 10, 5]
          dropout_probability  : (optional) dropout rate of the Fully Connected Layers. (default: 0.)


##############################
##### Input when forward #####
##############################

                        input  : A mini-batch of input data with categorical variables factorized in 
                                 the first half and numerical variables in the second half.
                                 The categorical variables need to follow the design at the time of model creation
                                 (number of variables, number of dictionaries for each categorical variable).


[1] Guo, Cheng, and Felix Berkhahn. "Entity embeddings of categorical variables." arXiv preprint arXiv:1604.06737 (2016).
'''


class CategoricalDnn(nn.Module):
    def __init__(self,
                 categorical_nuniques_to_dims,
                 num_numerical_features,
                 fc_layers_construction,
                 dropout_probability=0.):
        super(CategoricalDnn, self).__init__()

        self._make_embedding_layers(categorical_nuniques_to_dims)

        self._make_fc_layers(num_numerical_features, fc_layers_construction, dropout_probability)

    def _make_embedding_layers(self, nuniques_to_dims):
        """
        Define Embedding Layers of categorical variables.
         Properties:
             self.embedding_layer_list         :   List of Embedding Layers applied to categorical variable
             self.num_categorical_features     :   Total number of categorical variables before Embedding
             self.num_embedded_features        :   Total number of embedded features from categorical variables
             self.num_each_embedded_features   :   Number of embedded features for each categorical variable
        """
        self.num_categorical_features = len(nuniques_to_dims)
        self.num_embedded_features = 0
        self.num_each_embedded_features = []
        self.embedding_layer_list = nn.ModuleList()
        for nunique_to_dim in nuniques_to_dims:
            num_uniques = nunique_to_dim[0]
            target_dim = nunique_to_dim[1]
            self.embedding_layer_list.append(
                nn.Sequential(
                    nn.Embedding(num_uniques, target_dim),
                    nn.BatchNorm1d(target_dim),
                    # nn.ReLU(inplace=True)
                )
            )
            self.num_embedded_features += target_dim
            self.num_each_embedded_features.append(target_dim)

    def _make_fc_layers(self, num_numerical_features, fc_layers_construction, dropout_p):
        """
        Define input layer, hidden layer and output layer of the Fully Connected Layers.
         Properties:
             self.fc_layer_list   :   List of Fully Connected Layers that take embedded categorical features and
                                      numerical variables as inputs
             self.output_layer    :   Output layer
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
        # Split the input into categorical variables and numerical variables
        categorical_input = input[:, 0:self.num_categorical_features].long()
        numerical_input = input[:, self.num_categorical_features:]

        # Embed the categorical variables
        embedded = torch.zeros(input.size()[0], self.num_embedded_features)
        start_index = 0
        for i, emb_layer in enumerate(self.embedding_layer_list):
            gorl_index = start_index + self.num_each_embedded_features[i]
            embedded[:, start_index:gorl_index] = emb_layer(categorical_input[:, i])
            start_index = gorl_index

        # Concatenate the embedded categorical features and the numerical variables and pass it to FC layers
        out = torch.cat([embedded, numerical_input], axis=1)
        for hidden_layer in self.fc_layer_list:
            out = hidden_layer(out)
        return self.output_layer(out)



