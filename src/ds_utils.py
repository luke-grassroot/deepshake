import re
import subprocess
import os.path
import csv
import json
import gc
import syllables

from datetime import datetime
from torch.cuda import empty_cache

from datasets import load_dataset

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

#######################################
## APPROACH: CREATE NOISE-LINE PAIRS ##
#######################################

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


##########################################
## APPROACH: EXTRACT STRUCTURED SONNETS ##
##########################################

# writes in the format expected by HF dataset loader
def write_poems_to_file(file_path, all_poems, indices_to_write):
    with open(file_path, "w") as fout:
      [fout.write(json.dumps(all_poems[poem_index]) + "\n") for poem_index in indices_to_write]


def split_sonnets_into_stanzas(sonnet_file="../data/shakespeare-sonnets.txt", write_file_path=None):
    with open(sonnet_file, 'rt') as file:
      text = file.read()
      file.close()
    
    text = re.sub(r'[^\w\s]','',text)
    lines = text.split('\n')
    is_poem_start = lambda line: line.isupper()

    all_poems = []
    current_poem = []

    for line in lines[1:]:
        if is_poem_start(line):
            all_poems.append(current_poem)
            current_poem = []
        elif len(line.strip()) > 0:
            current_poem.append(line.strip())

    lengths = [len(sonnet) for sonnet in all_poems]
    word_lengths = [sum([len(line.split(' ')) for line in poem]) for poem in all_poems]
    
    # Now we want to break it out into stanzas
    structured_sonnets = []

    for poem in all_poems:
        structured_sonnet = { "text": "<newline>".join(poem) }
        
        stanzas = []
        current_stanza = []
        
        for i in range(len(poem)):
            current_stanza.append(poem[i])
            if (i + 1) % 4 == 0 or (i + 1) == len(poem):
                stanzas.append(current_stanza)
                current_stanza = []    
        
        structured_sonnet["stanzas"] = stanzas 
        structured_sonnets.append(structured_sonnet)
    
    number_sonnets = len(structured_sonnets)
    train_poems = [i for i in range(140) if i != lengths.index(min(lengths)) and i != lengths.index(max(lengths))]
    eval_poems = [i for i in range(number_sonnets) if i not in train_poems]

    if write_file_path is not None:
      write_poems_to_file(f"{write_file_path}_train.json", structured_sonnets, train_poems)
      write_poems_to_file(f"{write_file_path}_eval.json", structured_sonnets, eval_poems)

    return train_poems, eval_poems, lengths, word_lengths

def clean_up_speech(speech):
    character = re.compile("^[a-zA-Z]*:$")
    lines = speech.split("\n")
    actual_lines = [line for line in lines if not bool(character.match(line)) and len(line.strip()) > 0 and len(line.split()) > 3]
    speech_lines = [line for line in actual_lines if syllables.estimate(line) > 8]
    return speech_lines

def split_speech_into_stanzas(speech, stanza_length=4):
    stanzas = []
    current_stanza = []
    
    for i in range(len(speech)):
        current_stanza.append(speech[i])
        if (i + 1) % stanza_length == 0 or (i + 1) == len(speech):
            stanzas.append(current_stanza)
            current_stanza = []
    
    return stanzas

def split_play_text(play_text, min_speech_length=2):
  speeches = play_text.split("\n\n")
  cleaned_up_speeches = [clean_up_speech(speech) for speech in speeches]
  cleaned_up_speeches = [speech for speech in cleaned_up_speeches if len(speech) > 0]
  
  speeches_above_couplet = [speech for speech in cleaned_up_speeches if len(speech) >= min_speech_length]
  stanzas_deep = [split_speech_into_stanzas(speech) for speech in speeches_above_couplet]
  stanzas_shallow = [stanza for stanzas in stanzas_deep for stanza in stanzas if len(stanza) > 1]

  raw_speech_lengths = [len(speech) for speech in cleaned_up_speeches]
  # avg_stanza_length = sum([len(stanza) for stanza in stanzas_shallow])/len(stanzas_shallow)

  return stanzas_shallow, raw_speech_lengths
