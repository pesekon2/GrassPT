import argparse
import glob
import os
import re
import tempfile

import numpy as np

from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer,\
        GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments


class ConfigError(Exception):
    pass


# Functions to read different file types
def read_pdf(file_path: str) -> str:
    from PyPDF2 import PdfReader

    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()

    return text


def read_word(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'

    return text


def read_txt(file_path: str) -> str:
    with open(file_path, 'r') as file:
        text = file.read()

    return text


def parse_document(file_path: str, end_of_info: str = r'\n\n') -> str:
    if file_path.endswith('.pdf'):
        combined_text = read_pdf(file_path)
    elif file_path.endswith('.docx'):
        combined_text = read_word(file_path)
    elif file_path.endswith('.txt'):
        combined_text = read_txt(file_path)
    else:
        raise ConfigError(f'File {file_path} does not exist')

    combined_text = re.sub(end_of_info, '<|endoftext|>', combined_text)
    combined_text = combined_text.strip()  # Remove excess newline characters

    return combined_text


def parse_directory(dir_path: str) -> str:
    parsed_text = ''
    for file_path in glob.glob(os.path.join(dir_path, '*.txt')):
        read_text = read_txt(file_path)

        # GRASS module docs cleaning
        if 'Table of contents' in read_text:
            a = read_text.split('Table of contents')[0]
            if '  AUTHOR' in read_text:
                a += read_text.split('AUTHOR')[1]
            read_text = a
        if 'SEE ALSO' in read_text:
            parsed_text += read_text.split('SEE ALSO')[0]
        else:
            parsed_text += read_text

        parsed_text += '<|endoftext|>'

    return parsed_text.strip()  # Remove excess newline characters


def load_dataset(file_path: str, tokenizer, block_size: int = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size
    )

    return dataset


def load_data_collator(tokenizer, mlm: bool = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm
    )

    return data_collator

def train(train_file_path, model_name, output_dir, overwrite_output_dir,
          per_device_train_batch_size, num_train_epochs, save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a model.')

    parser.add_argument(
        '--nr_epochs', type=int, default=50,
        help='Number of epochs to train the model.'
    )

    parser.add_argument(
            '--training_data', type=str, default='training_data/q_and_a.txt',
        help='path to a file containing the training data or a directory '
             'containing training *.txt files'
    )

    args = parser.parse_args()

    end_of_info = r'\n\n'
    outdir = os.path.join('out', str(args.nr_epochs))
    model_name = 'gpt2'
    batch_size_per_device = 8
    save_steps = 50000
    overwrite_output_dir = True

    # Read the training file
    if os.path.isfile(args.training_data):
        text_data = parse_document(args.training_data, end_of_info)
    elif os.path.isdir(args.training_data):
        text_data = parse_directory(args.training_data)
    else:
        raise ConfigError(
            f'File or directory {args.training_data} does not exist'
        )

    # write it refactored as the training txt
    train_refactored_file = os.path.join(outdir, 'train.txt')
    if os.path.isdir(outdir) is False:
        os.makedirs(outdir)
    with open(train_refactored_file, 'w') as f:
        f.write(text_data)

    # train
    train(
        train_file_path=train_refactored_file,
        model_name=model_name,
        output_dir=outdir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=batch_size_per_device,
        num_train_epochs=args.nr_epochs,
        save_steps=save_steps
    )
