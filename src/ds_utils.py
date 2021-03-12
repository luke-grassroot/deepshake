import re
import subprocess
import os.path
import csv

from datetime import datetime

import gc
from torch.cuda import empty_cache

## Some basic utilities for GPU mgmt, saving, etc
# plundered from: https://github.com/huggingface/transformers/issues/1742
def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')    
    
# some model saving and loading helpers
def save_model(model, tokenizer, name):
    model.save_pretrained(f"results/models/{name}")
    tokenizer.save_pretrained(f"results/tokenizers/{name}")

def load_model(model_base, tokenizer_base, name, backup_tokenizer_name=None, device='cuda'):
    model = model_base.from_pretrained(f"results/models/{name}").to(device)
    try:
        tokenizer = tokenizer_base.from_pretrained(f"results/tokenizers/{name}/")
    except:
        print('Could not load tokenizer, HF bug still present, using default for model base')
        tokenizer = tokenizer_base.from_pretrained(backup_tokenizer_name)
    return model, tokenizer

# def reset_model(model):
#     del model
#     clean_up_gpu()
#     model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
#     return model

# reading in the sonnets
def read_poetry_into_lines(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    text = re.sub(r'[^\w\s]','',text)
    lines = [line.strip().split(' ') for line in text.split('\n') if len(line.strip()) > 0 and not (line.isupper())]
    return lines

# save some time by loading in or serializing a list of noise-output pairs
def read_dataset(number_pairs, lines_per_pair, tokenizer):
    file_path = f"./data/paired_noise_lines_{tokenizer.name_or_path}_{number_pairs}_{lines_per_pair}.csv"
    if not os.path.isfile(file_path):
        return False, None, None
    
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        combined = list(reader)
        inputs, labels = map(list, zip(*combined))
                
    return True, inputs, labels

def write_dataset(number_pairs, lines_per_pair, tokenizer, inputs, labels):
    input_path = f"./data/paired_noise_lines_{tokenizer.name_or_path}_{number_pairs}_{lines_per_pair}.csv"
        
    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'label'])
        writer.writerows(list(zip(inputs, labels)))

# serialize a generated list of poems, assumes a list of lists, each entry is a "poem", with lines in lists
def save_pseudo_poems(pseudo_poems, name=None):
    file_name = name if name else datetime.now().strftime("%Y_%m_%dT%H%M%S")
    file_path = f"results/written_lines/{file_name}"
    with open(file_path, "w") as output:
        for index, pseudo_poem in enumerate(pseudo_poems):
            out_text = f"{index+1}\n\n"
            for line in pseudo_poem:
                out_text += (str(line) + "\n")
            output.write(out_text + "\n\n")