# Program Overview
This program implements a poetry generation model based on RNN:
- **Model Construction**: Inherits nn.Module to implement the PoetryModel class.
- **Model Training**: Completes training of the PoetryModel based on a dataset of ancient poems.
- **Model Testing**: Implements test and acrostic_test functions to perform poem completion and acrostic poem generation.

## Class PoetryModel(nn.Module)
### \_\_init\_\_ Function
#### Overview
Defines the initialization function and mounts the necessary modules for training:
1. Define the model's hidden layer dimension, where `hidden_dim` is the input hidden layer dimension.
2. Use word embedding representation with `embedding_dim` as the input embedding vector dimension.
3. Define the LSTM model (using 2 layers of LSTM).
4. Define the linear model (mapping from `hidden_dim` to `vocab_size`).

#### Code Explanation
```python
super().__init__()
self.embedding_dim = embedding_dim
self.hidden_dim = hidden_dim
self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, 2)
self.output = nn.Linear(self.hidden_dim, vocab_size)
```

### forward() Function
#### Overview
Defines the forward propagation function:
- Input: `input`, with size (sequence_length, batch_size).
- Output: `output` (predictions) and `hidden` (hidden layer values).
  - `output` size: (sequence_length, batch_size, vocab_size)
  - `hidden` size: (4, batch_size, hidden_size)

#### Code Explanation
```python
seq_len, batch_size = input.size()
if hidden is None:
    h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
    c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
else:
    h_0, c_0 = hidden

input_embeddings = self.embeddings(input)
hiddens, hidden = self.rnn(input_embeddings, (h_0, c_0))
output = self.output(hiddens)

return output, hidden
```

## Testing
### Start_words - Should "<START>" be added?
Adding "<START>" improves the quality of generated poems by ensuring proper formatting. It helps the model predict more accurately.

### Input Error Handling
Two common input errors:
1. Omitting `.view(1,1)` can lead to a mismatch in expected and actual elements.
2. Missing `to(self.device)` can cause model and data mismatch when running on GPU.

### Acrostic_test() Model Training Insufficiency
Insufficient model training may lead to poor performance in generating acrostic poems.

## GPU Training Notes
### Training Time
Training time varies based on the dataset size and batch size. For example:
- With a batch size of 128 and 1000 poems: 6 minutes for 100 epochs.
- With a batch size of 128 and 50000 poems: 13 hours for 200 epochs.

### Training Speed
On GPU with a batch size of 128, training speed is approximately 0.5 seconds per batch.

## Sample Test Cases
- Sample poems for completion and acrostic poem generation are provided.
> 清风徐来花，夕色摇阴上。<br>
> 吞门得正棱，雪为菰翠午。<br>
> 芳创自自娱，幽物终有客。<br>
> 谁知公子来，亦有千里者。<br>
> 萧萧洒木泉，古木虫鸣宿。<br>
> 主人樵夫人，涕叟嗟我辱。<br>
> 顾我守渊市，得予事贱倒。<br>
> 人间我我师，我亦何况汝。<br>
> 况复哀子山，其也不敢酒。<br>
> 哀心在君日，犹是金石路。<br>
> 但觉雨声游，纵横抱新厅。<br>
> 芭苔谢秋中，寒食无一斑。<br>
> 此意若得谢，此意若可叹。<br>
---
> 清风徐来<br>
> 清光何可二，二十十二数。<br>
> 风如如我去，万古如不足。<br>
> 徐岂无故宫，岂可顾贪宅。<br>
> 来子不得其，一笑成绝迹。<br>



Feel free to adapt and use this README for your project documentation.
