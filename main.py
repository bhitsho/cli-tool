from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError
import os
import sys
import time
from dotenv import load_dotenv
import argparse
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ValidationError


# Constants
MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "analyze the sentiment of the user’s text"
BASE_URL = "https://api.groq.com/openai/v1"
MAX_ATTEMPTS = 3
BASE_DELAY=1

class KeyPhrase(BaseModel):
    phrase: str
    sentiment: Literal["positive", "negative", "mixed"]


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"]
    confidence_score: float = Field(ge=0, le=1)
    key_phrases: list[KeyPhrase]
    summary: str


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
        api_key=api_key,
        max_retries=0
    )

    analyze_with_retries(client,content)


def analyze(client, content, feedback= ""):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT+feedback},
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

def analyze_with_retries(client,content):
    RETRYABLE = (RateLimitError, APIConnectionError,
             APITimeoutError, InternalServerError)
    feedback = ""
    for attempt in range(MAX_ATTEMPTS):
        
        try:
            result = SentimentResult.model_validate_json(analyze(client, content, feedback))
            print(result.model_dump_json(indent=2))
            break
        
        except RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                sys.exit(
                    f"Error: failed after {MAX_ATTEMPTS} attempts ({type(e).__name__}: {e})")
            else:
                time.sleep(BASE_DELAY * (2**attempt))
                continue

        except ValidationError as e:
                lines = []
                for err in e.errors():
                    field = ".".join(str(part) for part in err["loc"])
                    lines.append(f"{field}: {err['msg']}")

                if attempt<MAX_ATTEMPTS-1:
                    feedback = "Your previous response failed validation.\n" + "\n".join(lines) + "\nCorrect these errors and return a valid JSON response."
                else:                    
                    sys.exit("Error: invalid model output\n" + "\n".join(lines))

        except APIError as e:
            sys.exit(f"Error: API request failed ({type(e).__name__}: {e})")

if __name__ == "__main__":
    main()
