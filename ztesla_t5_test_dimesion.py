__author__= "Amin Amiri"
__version__ = "v05312024"
#------[show header]-----
#load the CSV file
import pandas as pd

csv_path = './articles.csv'
data = pd.read_csv(csv_path)

#display the structure of the CSV file
print(data.head())
#-------[end]------------

# step 1: load the dataset
from datasets import load_dataset

#  use 'csv' to specify the format and provide the path to your local CSV file
dataset = load_dataset('csv', data_files='./articles.csv')

# check the dataset structure
print(dataset)

# step 2: Split the dataset into train and validation sets
# split the loaded dataset into training and validation sets with a 90-10 ratio
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']

#   step 3: preprocess the dataset for T5
from transformers import T5Tokenizer

#  load the tokenizer for the T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# function to preprocess the dataset
def preprocess_function(examples):
    # add the   "summarize: "   prefix to each text for the model to understand the task
    inputs = ["summarize: " + doc for doc in examples["text"]]
    # tokenize the inputs with a maximum length of 512 tokens and truncate if necessary
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    # tokenize the titles as targets with a maximum length of 128 tokens and truncate if necessary
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=128, truncation=True)
    # add the tokenized labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# apply the preprocessing function to the train and validation datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)

# ----------------------[step 4: Fine-tune the model]------------------------------------------------------
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# load the T5 model for conditional generation
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# define the training arguments for the Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir='./t5-title-generation',   # directory to save the model checkpoints and other outputs
    evaluation_strategy='epoch',          # evaluate the model at the end of each epoch
    learning_rate=2e-5,                   # learning rate for the optimizer
    per_device_train_batch_size=8,        # batch size for training
    per_device_eval_batch_size=8,         # batch size for evaluation
    weight_decay=0.01,                    # l2, weight decay for the optimizer
    save_total_limit=5,                   # limit the total number of saved checkpoints
    num_train_epochs=100,                   # number of training epochs
    predict_with_generate=True,           # use generate method for prediction
    fp16=True                             # use 16-bit (mixed) precision training
)

# create a data collator that will dynamically pad the inputs received, as well as the labels
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# initialize the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,                          # the  model to be trained
    args=training_args,                   # the training arguments defined above
    train_dataset=train_dataset,          # the training dataset
    eval_dataset=validation_dataset,      # the evaluation dataset
    data_collator=data_collator,          # the data collator for padding
    tokenizer=tokenizer                   # the tokenizer used for preprocessing
)

# start training the model
trainer.train()

print("\nThe program finished successfully.\n")
