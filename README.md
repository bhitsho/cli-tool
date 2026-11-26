# cli-tool-for-sentiment-analysis

This is the final version of the CLI tool for learning and testing llm api calls and their working. 

What this project basically does is classify the statement you provide into 4 labels, namely "positive", "negative", "mixed" and "neutral".It also provides a confidence score for how sure it is of that label along with some key phrases that led to the decision and at last a one-line summary of the text sentiment.

What I learned through this project was how to use OpenAI client to call api and provide it prompt and catch the output. I learned how to build a schema and force the api call to return the result as the schema I provided it. This helps one restricting the output to the type and format you want and prevent prompt injection by fixing the output format. I then learned to check if the provided output really does follow the schema I asked for the model to use or not. Lastly I implemented automatic retries for different situations, be it a model error, a user error, or a server error. I also implemented a feedback loop, wherein if the model gives out invalid output or output in invalid format, I send the error back as feedback along with the system prompt so that it may try again for a fixed number of attempts.

