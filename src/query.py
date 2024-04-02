import argparse
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
        # num_return_sequences=1
        # pad_token_id=tokenizer.eos_token_id
    )

    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ask query')

    parser.add_argument(
        '--query',
        type=str,
        default='How to get the manual for a certain module in GRASS GIS?',
        help='The query you would like to ask about GRASS GIS'
    )

    parser.add_argument(
        '--model', type=str,
        help='Path to the model to be used to answer'
    )

    args = parser.parse_args()

    max_len = 800
    generate_text(args.model, args.query, max_len)
