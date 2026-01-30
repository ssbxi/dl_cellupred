#import esm
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class LaccaseModel(nn.Module):
    def __init__(self, pretrained_model_path):
        super(LaccaseModel,self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.modelEsm = AutoModel.from_pretrained(pretrained_model_path)
        hidden_size = self.modelEsm.config.hidden_size
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    
        self._device = None
        
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.modelEsm.parameters()).device
        return self._device

    def forward(self, data, return_repr=False):
        out_result = self._get_representations(data) 
        out_put = self.dnn(out_result)
        
        if return_repr:
            return out_put, out_result
        else:
            return out_put
    
    def _get_layers(self):
        return self.modelEsm.config.num_hidden_layers
    
    @property
    def layers(self):
        return self.get_layers()
    
    def get_layers(self):
        return self._get_layers()
    
    def get_last_layer_idx(self):
        return self._get_layers()-1
    
    
    def _get_representations(self, data):
        sequences = [" ".join(list(seq.upper())) for name, seq in data]
        encoding = self.tokenizer(
                                sequences,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024,
                                
        )
       
        tokens = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        outputs = self.modelEsm(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state

        out_result = last_hidden_state[:, 0, :]
        return out_result
    
    def get_representations(self, data):
        return self._get_representations(data)
    
    def get_names(self, data):
        names = [name for name,seq in data]
        return names
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, state_dict_path=None):
        model = cls(pretrained_model_path)
        if state_dict_path is not None:
            print(f"Loading state dict from {state_dict_path}")
            model.load_state_dict(torch.load(state_dict_path))
        return model

        

