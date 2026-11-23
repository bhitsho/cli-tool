from openai import OpenAI
import os
import sys
import json
from dotenv import load_dotenv
import argparse
from pathlib import Path

# Constants
MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "analyze the sentiment of the user’s text"
BASE_URL = "https://api.groq.com/openai/v1"


def main():
    # loading environment variables
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

    result = json.loads(analyze(client, content))

    print(json.dumps(result, indent=2))


def analyze(client, content):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "mixed"]
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "key_phrases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "phrase": {
                                        "type": "string"
                                    },
                                    "sentiment": {
                                        "type": "string",
                                        "enum": ["positive", "negative", "mixed"]
                                    }
                                },
                                "required": ["phrase", "sentiment"],
                                "additionalProperties": False
                            }
                        },
                        "summary": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "sentiment",
                        "confidence_score",
                        "key_phrases",
                        "summary"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    main()
