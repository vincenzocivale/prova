### **Exercise 2 – Fine-tuning DistilBERT**

In this exercise, we perform a complete fine-tuning of the DistilBERT model for the sentiment classification task, going beyond the feature extraction approach used previously.


The dataset is tokenised using the DistilBERT tokeniser with active truncation. The `input_ids` and `attention_mask` fields, which are necessary for training, are obtained. The presence of all required fields (`text`, `label`, `input_ids`, `attention_mask`) in the tokenised datasets is also verified.
`AutoModelForSequenceClassification` is used, which adds a linear classification head (2 classes) to the DistilBERT model. The transformer weights are pre-trained, while the head is randomly initialised.

Training is handled by HuggingFace `Trainer`, with early stopping (patience = 3) and evaluation based on accuracy and F1-score. The main hyperparameters include:

* **Learning rate:** 2e-5
* **Batch size:** 64
* **Epochs:** up to 100
* **Weight decay:** 0.01

Training is end-to-end, updating both the transformer and the head. This approach improves the model's adaptation to the specific task and ensures better performance than simple feature extraction.