import os
import re
import argparse

import numpy as np

from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer,\
        GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments


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
    with open(file_path, "r") as file:
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
        raise Exception(f'File {file_path} does not exist')

    combined_text = re.sub(end_of_info, '<|endoftext|>', combined_text)
    combined_text = combined_text.strip()  # Remove excess newline characters

    return combined_text


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

    args = parser.parse_args()

    train_input_file = 'training_data/q_and_a.txt'
    end_of_info = r'\n\n'
    outdir = os.path.join('out', str(args.nr_epochs))
    model_name = 'gpt2'
    epochs_nr = 50
    batch_size_per_device = 8
    save_steps = 50000
    overwrite_output_dir = True

    # Read the training file
    text_data = parse_document(train_input_file, end_of_info)

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
