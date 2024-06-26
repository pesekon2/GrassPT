import argparse
import glob
import os
import re
import tempfile

import numpy as np

from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer,\
        GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments


class ConfigError(Exception):
    """Exception to be raised if the config of the model is invalid."""
    pass


# Functions to read different file types
def read_pdf(file_path: str) -> str:
    """Read a .pdf file.

    In order to read a .pdf file, the Python package PyPDF2 has to be installed

    :param file_path: path to the file
    :return: string with the contents of the file
    """
    from PyPDF2 import PdfReader

    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()

    return text


def read_word(file_path: str) -> str:
    """Read a .docx file.

    In order to read a .docx file, the Python package docx has to be installed

    :param file_path: path to the file
    :return: string with the contents of the file
    """
    import docx

    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'

    return text


def read_txt(file_path: str) -> str:
    """Read a .txt file.

    :param file_path: path to the file
    :return: string with the contents of the file
    """
    with open(file_path, 'r') as file:
        text = file.read()

    return text


def parse_document(file_path: str, end_of_info: str = r'\n\n') -> str:
    """Parse a single file to get a training data string.

    :param file_path: path to a file containg the training data
    :param end_of_info: every end_of_info occurence is considered the end of an
        information chunk and is therefore replaced with '<|endoftext|>'
    :return: string containg all the training data information separated by
        <|endoftext|>
    """
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
    """Parse all *.txt files in directory to get a single training data string.

    '<|endoftext|>' is added to the end of every file to separate their
    contents. ALso, if 'Table of contents' is present in a file, it is
    considered a GRASS GIS module manual and sections Table of contents,
    AUTHOR, SEE ALSO, and SOURCE CODE are deleted from the read file.

    :param dir_path: path to a directory containg training *.txt files
    :return: string containg all the training data information separated by
        <|endoftext|>
    """
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
    """Load dataset as a TextDataset object.

    :param file_path: path to the file including training data
    :param tokenizer: an object performing the tokenization (the process of
        converting a sequence of text into tokens, i.e. smaller parts)
    :param block_size: size of individual blocks in the dataset
    :return: a TextDataset object
    """
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size
    )

    return dataset


def load_data_collator(tokenizer, mlm: bool = False):
    """Load the data collator.

    Data collators are objects that will form a batch by using a list of dataset
    elements as input.

    :param tokenizer: an object performing the tokenization (the process of
        converting a sequence of text into tokens, i.e. smaller parts)
    :param mlm: boolean saying whether to use masked language modeling
    :return: data collator object
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm
    )

    return data_collator

def train(train_file_path: str, output_dir: str,
          overwrite_output_dir: bool = False, model_name: str = 'gpt2',
          per_device_train_batch_size: int = 8, num_train_epochs: int = 50,
          save_steps: int = 50000) -> None:
    """Train a model.

    :param train_file_path: path to a filename containing training data.
        Sectors or chunks of information should be separated by '<|endoftext|>'
    :param output_dir: directory where the output model will be written
    :param overwrite_output_dir: boolean saying whether to overwrite the output
        directory or not
    :param model_name: name of a pretrained model to be used
    :param per_device_train_batch_size: batch size per one computation device
    :param num_train_epochs: number of training epochs
    :param save_steps: after how many steps should be checkpoint models written
    """
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
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
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
        output_dir=outdir,
        overwrite_output_dir=overwrite_output_dir,
        model_name=model_name,
        per_device_train_batch_size=batch_size_per_device,
        num_train_epochs=args.nr_epochs,
        save_steps=save_steps
    )
