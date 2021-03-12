import gc
import random
from math import floor, ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ds_utils import show_gpu, read_dataset, write_dataset, save_model, load_model, read_poetry_into_lines, save_pseudo_poems

device = 'cuda'

# could probably make global but one of the cheapest operations all round, so might as well
shake_sonnet_lines = read_poetry_into_lines('./data/shakespeare-sonnets.txt')
max_line_length = max([len(line) for line in shake_sonnet_lines])

def clean_up_gpu():
    show_gpu('Preclean: ')
    torch.cuda.empty_cache()
    gc.collect()
    show_gpu('Postclean: ')

def assemble_paired_noise_text(tokenizer, number_pairs, lines_per_pair=1, input_length=10, command="shakespeare", add_final_eos_token=False):
    print('Number of lines: ', len(shake_sonnet_lines), " and max length: ", max_line_length)

    random_set = [random.sample(list(tokenizer.vocab.keys()), input_length) for _ in range(number_pairs)] 
    generated_noise = [f"{command}: {' '.join(random_words)}" for random_words in random_set]
    
    actual_lines = []

    max_sample = len(shake_sonnet_lines) / lines_per_pair
    number_loops = ceil(number_pairs / max_sample)

    # we randomly choose a number of starting points, and then pick from their forwards
    starting_points = random.choices(range(floor(max_sample)), k=number_pairs)
    for i in range(number_pairs):
        initial_line = starting_points[i]
        lines_to_use = shake_sonnet_lines[initial_line:initial_line+lines_per_pair]
        consolidated_line = tokenizer.eos_token.join([' '.join(line) for line in lines_to_use])
        if add_final_eos_token:
            consolidated_line += tokenizer.eos_token
        actual_lines += [consolidated_line]
        
    return generated_noise, actual_lines 

# make deepshake actually output
def generate_lines(model, tokenizer, input_length=10, print_output=True, skip_special_tokens=False, print_input=False, generator_args={}):
    random_words = random.sample(list(tokenizer.vocab.keys()), input_length)
    tokenized_random = tokenizer(f"generate Shakespeare: {' '.join(random_words)}", padding=True, return_tensors="pt").to(device)
    model_output = model.generate(**tokenized_random, max_length=max_line_length, **generator_args)
    generated_line = tokenizer.decode(model_output[0], skip_special_tokens=skip_special_tokens)
    if print_input:
        print('Random words: ', random_words, ' and output: ', generated_line)
    elif print_output:
        print(generated_line)
    return generated_line

def write_sonnet(model, tokenizer, generator_args=None, print_sonnet=False):
    model_gen_args=generator_args if generator_args else {}
    lines = [generate_lines(model, tokenizer, print_output=print_sonnet, skip_special_tokens=True, print_input=False, generator_args=model_gen_args) for l in range(14)]
    if print_sonnet:
        print('\n'.join(lines))
    return lines

def wrte_and_save_sonnet(model, tokenizer, name, generator_args=None):
    sonnets = [write_sonnet(model, tokenizer, generator_args) for s in range(10)]
    save_pseudo_poems(sonnets, name)
    return sonnets

# HF utilities for dataset and collation
class NoiseToShakeDataset(torch.utils.data.Dataset):
    def __init__(self, generated_noise, paired_lines):
        self.generated_noise = generated_noise
        self.sonnet_lines = paired_lines
    
    def __getitem__(self, idx):
        return { 'tgt_texts': self.sonnet_lines[idx], 'src_texts': self.generated_noise[idx], 'id': idx }
    
    def __len__(self):
        return len(self.generated_noise)
    
    def collate_fn(self, batch):
        batch_encoding = tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tg_texts"] for x in batch],
            return_tensors="pt"
        )
        
class NoiseToShakeDataCollator:
    def __init__(self, tokenizer, data_args=None):
        self.data_args = data_args
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        batch = self._encode(batch)
        input_ids, attention_mask, labels = (
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        return batch
    
    def _encode(self, batch):
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            padding="max_length",
            max_length=max_line_length,
            return_tensors="pt"
        )
        return batch_encoding
    

# handy thing to see what's happening
class PrintExampleCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        generate_lines(model, tokenizer)
        
        
# generate the dataset needed for training (and/or write and read if necessary)
def assemble_dataset(tokenizer, number_pairs, lines_per_pair=2, input_length=10, check_for_saved=False, write_generated=False, tokenizer_name=None):
    loaded_prior = False
    if check_for_saved:
        loaded_prior, inputs, labels = read_dataset(number_pairs, lines_per_pair, tokenizer)
        
    if not loaded_prior:
        inputs, labels = assemble_paired_noise_text(tokenizer, input_length=input_length, number_pairs=number_pairs, lines_per_pair=lines_per_pair)
    
    if write_generated and not loaded_prior:
        write_dataset(number_pairs, lines_per_pair, tokenizer, inputs, labels)
    
    print('Assembled dataset, first input: ', inputs[0], ' and first label: ', labels[0])
    return NoiseToShakeDataset(inputs, labels)

        
# main method
def experiment(
    label,
    model_base, 
    tokenizer_base, 
    pretrained_name,
    custom_training_args,
    number_training_pairs=10000,
    lines_per_pair=2,
    number_validation_pairs=200,
    input_length=5,
    generator_args={},
    add_eos_token_to_labels=False, 
    verbose=False,
    save_model=False
):
    model = model_base.from_pretrained(pretrained_name).to(device)
    tokenizer = tokenizer_base.from_pretrained(pretrained_name)
    
    clean_up_gpu()
    show_gpu('Loaded model, current GP use:')
    
    if verbose:
        generate_lines(model, tokenizer)
        noise, line = assemble_paired_noise_text(tokenizer, add_eos_token=add_eos_token_to_labels)
        print(noise, line)
        print(NoiseToShakeDataset(noise, line)[0])
    
    train_dataset = assemble_dataset(tokenizer, number_pairs=number_training_pairs, lines_per_pair=2, check_for_saved=True, write_generated=True)
    val_dataset = assemble_dataset(tokenizer, number_pairs=number_validation_pairs, lines_per_pair=2, check_for_saved=True, write_generated=True)
    
    clean_up_gpu()
    show_gpu('After composing datasets: ')
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',          
        per_device_train_batch_size=128,
        logging_dir='./logs',
        logging_steps=20,
        **custom_training_args
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=NoiseToShakeDataCollator(tokenizer=tokenizer),
        callbacks=[PrintExampleCallback()]
    )
        
    trainer.train()
    
    if verbose:
        output_once_off(model, tokenizer, input_length)
    
    sonnets = wrte_and_save_sonnet(model, tokenizer, label, generator_args)
    
    if save_model:
        save_this_model(model, tokenizer, 't5_base_20k_1line')
    
    return sonnets