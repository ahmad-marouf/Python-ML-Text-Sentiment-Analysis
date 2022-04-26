import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class IMDBReviewsDataset(Dataset):

  def __init__(self, reviews, sentiments, tokenizer, max_len):
    self.reviews = reviews
    self.sentiments = sentiments
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    sentiment = self.sentiments[item]
    encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = self.max_len,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_token_type_ids = False,
        return_tensors = 'pt',  # Return PyTorch tensors
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(sentiment, dtype=torch.long)

    }

def create_data_loader(features, classifications, tokenizer, max_len, batch_size):
    dataset = IMDBReviewsDataset(
        reviews = features.to_numpy(),
        sentiments = classifications.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 4
    )