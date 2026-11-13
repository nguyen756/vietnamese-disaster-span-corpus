"""Three heads sharing exactly the same original-unit PhoBERT pooling."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class NERModel(nn.Module):
    def __init__(self, encoder, labels, architecture, hidden=128, dropout=0.2):
        super().__init__()
        assert architecture in ('phobert', 'bilstm', 'bilstm_crf')
        self.encoder = encoder
        self.labels = list(labels)
        self.architecture = architecture
        self.dropout = nn.Dropout(dropout)
        width = encoder.config.hidden_size
        if architecture != 'phobert':
            self.sequence = nn.LSTM(width, hidden, batch_first=True, bidirectional=True)
            width = hidden * 2
        self.classifier = nn.Linear(width, len(labels))
        if architecture == 'bilstm_crf':
            self.crf = CRF(len(labels), batch_first=True)

    def forward(self, batch):
        hidden = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
        flat = hidden.reshape(-1, hidden.shape[-1])
        selected = flat.index_select(0, batch['pool_src']) * batch['pool_weight'].unsqueeze(1)
        units = hidden.new_zeros((batch['batch_size'] * batch['max_units'], hidden.shape[-1]))
        units.index_add_(0, batch['pool_dst'], selected)
        units = units.reshape(batch['batch_size'], batch['max_units'], hidden.shape[-1])
        if self.architecture != 'phobert':
            packed = pack_padded_sequence(self.dropout(units), batch['lengths'], batch_first=True, enforce_sorted=False)
            output, _ = self.sequence(packed)
            units, _ = pad_packed_sequence(output, batch_first=True, total_length=batch['max_units'])
        return self.classifier(self.dropout(units))

    @staticmethod
    def mask(batch, device):
        return torch.arange(batch['max_units'], device=device)[None, :] < batch['lengths'].to(device)[:, None]

    def loss(self, emissions, batch):
        if self.architecture == 'bilstm_crf':
            mask = self.mask(batch, emissions.device)
            tags = batch['targets'].masked_fill(~mask, 0)
            # Match cross-entropy's token-mean reduction, including variable lengths.
            return -self.crf(emissions, tags, mask=mask, reduction='token_mean')
        return nn.functional.cross_entropy(emissions.reshape(-1, len(self.labels)), batch['targets'].reshape(-1), ignore_index=-100)

    def tag_sequences(self, emissions, batch):
        if self.architecture == 'bilstm_crf':
            return self.crf.decode(emissions, mask=self.mask(batch, emissions.device))
        tags = emissions.argmax(-1).cpu().tolist()
        return [row[:int(length)] for row, length in zip(tags, batch['lengths'])]
