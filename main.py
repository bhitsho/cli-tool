from openai import OpenAI
import os, sys
from dotenv import load_dotenv
import argparse
from pathlib import Path

#Constants
MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "You are a helpful assistant."
BASE_URL = "https://api.groq.com/openai/v1"

def main():
    #loading environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Text processing CLI tool")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("text", type=str, nargs="?", help="Input text")
    group.add_argument("-f", "--file", type=Path, help="Path to input file")

    args = parser.parse_args()

    if args.file is not None:
        try:
            content = args.file.read_text(encoding="utf-8")

        except FileNotFoundError:
            parser.error("File does not exist")

        except UnicodeDecodeError:
            parser.error("File must be UTF-8 encoded")

        except PermissionError:
            parser.error("Permission denied")

        except OSError as e:
            parser.error(f"Unable to read file: {e}")

    else:
        content = args.text

    if not content.strip():
        parser.error("Input cannot be empty")

    api_key = os.environ.get("GROK_API_KEY")

    if not api_key or not api_key.strip():
        sys.exit("Error: API key is missing or empty")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key
    )

    result = analyze(client,content)

    print(result)


def analyze(client,content):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    main()
