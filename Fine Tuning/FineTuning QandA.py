import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer
import os


base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_link= "mlabonne/guanaco-llama2-1k"
new_model="llama-1.1B-chat-guanaco" #name of our new model
ds = load_dataset(dataset_link,split='train')

#print(ds[0])
quantization_config=BitsAndBytesConfig(load_in_4bit=True)
model=AutoModelForCausalLM.from_pretrained(base_model,device_map='auto',torch_dtype=torch.float16) #device map auto means hugging face will automatically use our GPU
model.config.use_cache=False
model.config.pretraining_tp=1 #no parallel computing
tokenizer =AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token #pad_sequence
tokenizer.padding_side='right'

#run inference
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is Napoleon Bonaparte?"
pipe= pipeline(task='text-generation',model=model, tokenizer=tokenizer, max_length=100)
result=pipe(f"{prompt}")
print(result[0]['generated_text'])

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
tokenized_dataset = ds.map(preprocess_function, batched=True)

os.environ["WANDB_DISABLED"] = "true"
peft_params=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
training_params = TrainingArguments(output_dir='.\results',
                                    num_train_epochs=2,
                                    per_device_train_batch_size=2,
                                    gradient_accumulation_steps=16,
                                    optim='adamw_torch',
                                    save_steps=25,
                                    logging_steps=1,
                                    learning_rate=2e-4,
                                    weight_decay=0.001,
                                    fp16=True,
                                    bf16=False,
                                    max_grad_norm=0.3,
                                    max_steps=-1,
                                    warmup_ratio=0.03,
                                    group_by_length=True,
                                    lr_scheduler_type='cosine'
                                    )

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_params,
    args=training_params)
import gc
gc.collect()
torch.cuda.empty_cache()

trainer.train()
trainer.save_model(new_model)
trainer.tokenizer.save_pretrained(new_model)

#run updated model
prompt = "Who is Napoleon Bonaparte?"
pipe= pipeline(task='text-generation',model=model, tokenizer=tokenizer, max_length=100)
result=pipe(f"{prompt}")
print(result[0]['generated_text'])