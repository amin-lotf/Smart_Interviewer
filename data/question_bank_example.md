#Level 1 — Basic LLM Fundamentals
##Item: LLM-definition
context:
LLM stands for Large Language Model. It is a type of artificial intelligence model trained on very large collections of text data.
The goal of an LLM is to learn patterns in language so it can predict the next token (word or subword) given a sequence of previous tokens.
LLMs do not understand language like humans do; instead, they rely on statistical patterns learned during training.
Common applications of LLMs include question answering, text generation, summarization, and translation.

objective:
Verify that the user knows what the acronym LLM stands for and has a basic idea of what it is.



##Item: LLM-purpose
context:
The main purpose of a Large Language Model is to generate and understand text by predicting what comes next in a sequence.
During training, the model is exposed to massive amounts of text and learns how words and phrases commonly appear together.
Once trained, an LLM can be used to answer questions, continue a sentence, write explanations, or summarize information.
It does not search the internet or retrieve facts by default; it generates responses based on patterns learned during training.

objective:
Check whether the user understands what an LLM is designed to do at a high level.






##Item: Tokens-basic
context:
Language models do not process text as full sentences or whole words directly. Instead, they break text into smaller units called tokens.
A token can be a whole word, part of a word, or sometimes a punctuation mark.
For example, a long or uncommon word may be split into multiple tokens.
The model predicts one token at a time, and this token-by-token prediction is how text is generated.

objective:
Ensure the user understands what tokens are in simple terms.



##Item: Training-data
context:
Large Language Models are trained on very large datasets that contain text from many sources, such as books, articles, and websites.
During training, the model repeatedly tries to predict the next token in a sentence and adjusts its internal parameters when it makes mistakes.
This process allows the model to learn grammar, common facts, and general language patterns.
However, the model does not remember individual documents; it only retains statistical patterns from the data.

objective:
Check if the user understands, at a basic level, how LLMs are trained.


##Item: No-real-understanding

context:
Although LLMs can produce very fluent and convincing text, they do not have true understanding or consciousness.
They do not know whether an answer is true or false in a human sense unless guided by training or evaluation mechanisms.
Their responses are generated based on probabilities, not reasoning or awareness.
This is why LLMs can sometimes produce confident but incorrect answers.

objective:
Confirm that the user understands the limitations of LLMs at a basic conceptual level.



#Level 2 — Core LLM Concepts (Conceptual)
##Item: Next-token-prediction
context:
At the core of a Large Language Model is the task of next-token prediction.
Given a sequence of tokens, the model estimates the probability of what the next token should be.
By repeatedly predicting one token at a time, the model can generate full sentences, paragraphs, or longer text.
This simple mechanism, when trained on massive datasets, is what enables LLMs to perform many complex language tasks.

objective:
Check whether the user understands the fundamental mechanism behind text generation.


##Item: Probability-and-uncertainty

context:
LLMs do not produce a single guaranteed output for a given input.
Instead, they assign probabilities to many possible next tokens.
Depending on settings such as randomness or temperature, the model may choose more predictable or more creative outputs.
This probabilistic nature explains why the same prompt can produce different responses.

objective:
Ensure the user understands that LLM outputs are probabilistic, not deterministic.




##Item: Context-window
context:
LLMs can only consider a limited amount of text at one time, known as the context window.
The context window includes the prompt and any prior conversation or text provided to the model.
If information falls outside this window, the model cannot directly use it when generating a response.
This limitation is one reason techniques like summarization or retrieval are used in real applications.

objective:
Verify that the user understands what a context window is and why it matters.



##Item: Hallucination-basic
context:
A hallucination occurs when an LLM generates information that sounds correct but is actually false or made up.
This happens because the model prioritizes producing fluent text rather than verifying facts.
Hallucinations are more likely when the model is asked about topics it has limited or unclear information about.
Reducing hallucinations often requires grounding the model with external data or constraints.

objective:
Check whether the user understands what hallucinations are and why they happen.




##Item: Prompt-role
context:
A prompt is the input text given to an LLM to guide its response.
The wording, clarity, and structure of a prompt can strongly influence the quality of the output.
Clear prompts with sufficient context usually lead to better answers.
This is why prompt design is an important practical skill when working with LLMs.

objective:
Confirm the user understands the role of prompts in guiding model behavior.

