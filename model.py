# import torch and layers
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, BCELoss, Sigmoid, BatchNorm1d

# and huggingface
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# ok ok new model
class Model(torch.nn.Module):

    def __init__(self, base_model, in_features, out_features=1):
        # initalize
        super().__init__()

        # create base model
        model = BertModel.from_pretrained(base_model)
        self.base_model = model

        # meta feature norm
        self.meta_feature_norm = BatchNorm1d(in_features)
        # create input embedding
        self.meta_feature_embedding = Linear(in_features, model.config.hidden_size)

        # create output layer
        self.out = Linear(model.config.hidden_size, out_features)

        # sigmoid
        self.sigmoid = Sigmoid()

        # loss function
        self.bce_loss = BCELoss()

    # forward
    def forward(self, meta_features, labels=None, **kwargs):
        # pass kwargs into the model
        base_out = self.base_model(**kwargs)
        # norm
        meta_normed = self.meta_feature_norm(meta_features)
        # input metafeature enmebdding
        meta_embedding = self.meta_feature_embedding(meta_normed)
        # late fusion
        fusion = base_out["pooler_output"] + meta_embedding
        # output
        output = self.sigmoid(self.out(fusion))

        # if training, calculate and return loss
        if self.training and labels != None:
            loss = self.bce_loss(output, labels)
            return {"logits": output, "loss": loss}
        # if not, just return output
        else:
            return {"logits": output}


