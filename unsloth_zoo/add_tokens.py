# python add_tokens.py --model Qwen/Qwen2.5-1.5B --outputDirectory ~/qwen-1.5-llama3 --template llama3

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from tokenizer_utils import add_new_tokens, NewToken

DEFAULT_INTERPOLATION = 0.5

NEW_TOKENS_GEMMA = [
    NewToken(
        label="<start_of_turn>",
        initial_embedding=[(" start", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<end_of_turn>",
        initial_embedding=[(" end", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_start|>",
        initial_embedding=[(" start", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_end|>",
        initial_embedding=[(" end", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
]

NEW_TOKENS_LLAMA3 = [
    NewToken(
        label="<|start_header_id|>",
        initial_embedding=[(" start", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|end_header_id|>",
        initial_embedding=[(" end", 0.5), (" message header", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|eot_id|>",
        initial_embedding=[(" end", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_start|>",
        initial_embedding=[(" start", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_end|>",
        initial_embedding=[(" end", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
]

NEW_TOKENS_LLAMA3_QWQ = [
    NewToken(
        label="<|end_header_id|>",
        initial_embedding=[("\n", 1)],
        initial_embedding_interpolation=0.9,
    ),
    # We rename the existing tokens instead.
    # NewToken(
    #     label="<|start_header_id|>",
    #     initial_embedding=[("<|im_start|>", 1)],
    #     initial_embedding_interpolation=0.8,
    # ),
    # NewToken(
    #     label="<|eot_id|>",
    #     initial_embedding=[("<|im_end|>", 0.8), ("\n", 0.2)],
    #     initial_embedding_interpolation=0.8,
    # ),
    # NewToken(
    #     label="<|reasoning_start|>",
    #     initial_embedding=[("<think>", 0.9), ("\n", 0.1)],
    #     initial_embedding_interpolation=0.8,
    # ),
    # NewToken(
    #     label="<|reasoning_end|>",
    #     initial_embedding=[("</think>", 1)],
    #     initial_embedding_interpolation=0.8,
    # ),
]

NEW_TOKENS_CHATML = [
    NewToken(
        label="<|im_start|>",
        initial_embedding=[(" start", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|im_end|>",
        initial_embedding=[(" end", 0.5), (" message", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_start|>",
        initial_embedding=[(" start", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
    NewToken(
        label="<|reasoning_end|>",
        initial_embedding=[(" end", 0.5), (" thinking", 0.5)],
        initial_embedding_interpolation=DEFAULT_INTERPOLATION,
    ),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add new tokens to a model and tokenizer"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--outputDirectory",
        type=str,
        required=True,
        help="Directory to save the model and tokenizer",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["llama3", "chatml", "llama3_qwq", "gemma"],
        required=True,
        help="Template to use for new tokens",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Select tokens based on template
    if args.template.lower() == "llama3":
        new_tokens = NEW_TOKENS_LLAMA3
    elif args.template.lower() == "chatml":
        new_tokens = NEW_TOKENS_CHATML
    elif args.template.lower() == "llama3_qwq":
        new_tokens = NEW_TOKENS_LLAMA3_QWQ
    elif args.template.lower() == "gemma":
        new_tokens = NEW_TOKENS_GEMMA
    else:
        raise RuntimeError(f"Unknown template {args.template}")

    # Create output directory if it doesn't exist
    os.makedirs(args.outputDirectory, exist_ok=True)

    # Add new tokens
    add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=new_tokens,
        method="interpolation",
    )

    # Save model and tokenizer
    model.save_pretrained(args.outputDirectory)
    tokenizer.save_pretrained(args.outputDirectory)

    print(f"Model and tokenizer with new tokens saved to {args.outputDirectory}")
