# import torch and layers
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, BCELoss, Dropout, Softmax,BatchNorm1d

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# ok ok new model
class Model(torch.nn.Module):

    def __init__(self, base_model, in_features, out_features=2, hidden_features=128):
        # initalize
        super().__init__()

        # create base model
        model = BertForSequenceClassification.from_pretrained(base_model)
        self.base_model = model
        self.model_droupout = Dropout(p=0.1, inplace=False)

        # meta feature norm
        self.meta_feature_norm = BatchNorm1d(in_features)
        # create input embedding
        self.meta_feature_embedding_0 = Linear(in_features, model.config.hidden_size)
        self.meta_feature_embedding_1 = Linear(model.config.hidden_size,
                                               model.config.hidden_size)
        # meta droupout
        self.meta_feature_droupout = Dropout(p=0.1, inplace=False)

        # create output layer
        self.out = Linear(model.config.hidden_size, out_features)

        # sigmoid
        self.softmax = Softmax(dim=1)

        # loss function
        self.bce_loss = BCELoss()

    # forward
    def forward(self, meta_features, labels=None, **kwargs):
        # pass kwargs into the model
        base_out = self.base_model(**kwargs)
        # norm
        meta_normed = self.meta_feature_norm(meta_features)

        # input metafeature enmebdding
        meta_embedding = F.relu(self.meta_feature_embedding_0(meta_features))
        meta_embedding = F.relu(self.meta_feature_embedding_1(meta_embedding))
        meta_embedding = self.out(self.meta_feature_droupout(meta_embedding))

        # output
        output = self.softmax(base_out["logits"]+meta_embedding)

        # if training, calculate and return loss
        if self.training and labels != None:
            loss = self.bce_loss(output, labels)
            return {"logits": output, "loss": loss}
        # if not, just return output
        else:
            return {"logits": output}


