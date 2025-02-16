import torch
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging,DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import evaluate
from transformers import GenerationConfig
import pandas as pd

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
ds = load_dataset("knkarthick/dialogsum")
model_name= 'sshleifer/distilbart-cnn-12-6'#facebook/bart-large-cnn
#model_name = AutoModelForCausalLM.from_pretrained(model=model_name, quantization_config=bnb_config, device_map="auto")
model_4bit= AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
#Testing Base model
index=5
prompt=ds['test'][index]['dialogue']
summary=ds['test'][index]['summary']
summarizer = pipeline(task="summarization", model=model_name,  tokenizer=tokenizer)
result = summarizer(prompt, max_length=200)
print(f"Input {prompt}")
dash='-'
dash_line = dash * 100
print(dash_line)
print(f"Baseline summary: {summary}")
print(dash_line)
print("Model Summary: ",result[0]['summary_text'])

#Preprocessing of data
def tokenizefunc (example):
  start_prompt ='Summarize the following conversation.\n'
  end_prompt = '\n\nSummary: '
  #prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dailogue"]]
  prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
  example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,return_tensors="pt").input_ids
  example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True,return_tensors="pt").input_ids

  return example

tokenized_dataset = ds.map(tokenizefunc, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['id', 'topic', 'dialogue', 'summary',])
tokenized_dataset = tokenized_dataset.filter(lambda example, index: index % 100 == 0, with_indices=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name,padding=True)

#Setting parameters for training
os.environ['WANDB_DISABLED']="true"
original_model = prepare_model_for_kbit_training(model_4bit)
config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(output_dir='./results',
                                    num_train_epochs=5,
                                    per_device_train_batch_size=8,
                                    per_device_eval_batch_size=8,
                                    logging_dir="./logs",
                                    eval_strategy="steps",
                                    fp16=True,
                                    )
peft_model = get_peft_model(original_model, config)

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    args=training_params,
    data_collator=data_collator,
)

trainer.train()


trained_model_dir = "./trained_model"
trainer.save_model(trained_model_dir)

# Load the trained model
trained_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_dir)
#Testing trained model
index=5
prompt=ds['test'][index]['dialogue']
summary=ds['test'][index]['summary']
summarizer = pipeline(task="summarization", model=trained_model,  tokenizer=tokenizer)
result = summarizer(prompt, max_length=200)
print(f"Input {prompt}")
dash='-'
dash_line = dash * 100
print(dash_line)
print(f"Baseline summary: {summary}")
print(dash_line)
print("Model Summary: ",result[0]['summary_text'])

#Viewing inprovement in results after finetuning
dialogues = ds['test'][0:10]['dialogue']
human_baseline_summaries = ds['test'][0:10]['summary']
original_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
  prompt = f"""
  summarize the following conversation
  {dialogue}
  Summary:
  """
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  # Ensure that input_ids and the models are on the same device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_ids = input_ids.to(device)
  human_baseline_text_output = human_baseline_summaries[idx]
  original_model_outputs = model_4bit.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
  original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
  original_model_summaries.append(original_model_text_output)
  peft_model_outputs = trained_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
  peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
  peft_model_summaries.append(peft_model_text_output)
zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
print(df.head())

#Calculating Rouge score
rouge = evaluate.load('rouge')
original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
print('ORIGINAL MODEL:')
print(original_model_results)

print('PEFT MODEL:')
print(peft_model_results)

#Calculating precentage improvement after finetuning

import numpy as np
print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')