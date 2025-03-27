# CLAUDE.md: Repository Guidelines

## Build & Run Commands
- **Core Installation**: `pip install -r requirements.txt`
- **Basic Run**: `python cli.py example-prompts.yaml` (prints results as they arrive)
- **Save Output to File**: `python cli.py example-prompts.yaml -o results.json` (prints results and saves to file)
- **Hide Terminal Output**: `python cli.py example-prompts.yaml -o results.json --silent` (only saves to file)
- **Print Specific Results**: `python cli.py example-prompts.yaml --print summary critique` (only prints specified prompts)
- **Show Prompts**: `python cli.py example-prompts.yaml --show-prompts` (shows original and resolved prompts with results)
- **Batch Mode**: `python cli.py example-prompts.yaml --batch` (waits for all prompts to complete before displaying results)
- **Debug Mode**: `python cli.py example-prompts.yaml --debug` (shows configs and more info about LLM results)
- **Verbose Debug**: `python cli.py example-prompts.yaml --debug -v` (displays full prompts and detailed configs)
- **Model Listing**: `python cli.py --provider openai --list-models`
- **Validate Models**: `python cli.py example-prompts.yaml --validate-models`
- **Use with Context**: `python cli.py example-prompts.yaml --context background:context1.txt examples:context2.txt`
- **Direct Prompt**: `python cli.py "Explain quantum computing in simple terms"`
- **Streaming Output**: `python cli.py -p openai -m gpt-4o --stream "Tell me a story"` (shows response as it's generated, direct prompts only)
- **Gemini Example**: `python cli.py -p gemini -m gemini-1.5-flash "Explain machine learning"`

## Output Format
- **Default Mode**: Shows results immediately as they arrive from LLMs
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: intro ==

  [Output text...]

  ==================================================

  == RESULT: summary ==

  [Output text...]

  ==================================================
  ```

- **Streaming Mode**: Shows response tokens as they are generated (direct prompts only)
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: direct_prompt ==

  Once upon a time, in a distant galaxy, there was a small robot who loved to tell stories...
  ```

- **Batch Mode**: Shows all results after all prompts complete
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: intro ==

  [Output text...]

  ==================================================

  == RESULT: summary ==

  [Output text...]

  ==================================================
  ```

- **With Show Prompts**: Shows prompts and results
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: intro ==

  Prompt:
  Explain OOP in simple terms.

  Response:
  [Output text...]

  ==================================================
  ```

- **With Resolved Prompts**: Shows original and resolved prompts for dependent prompts
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: summary ==

  Original Prompt:
  Summarize the following: {{ intro }}

  Resolved Prompt:
  Summarize the following: Object-Oriented Programming (OOP) is a programming approach...

  Response:
  [Output text...]

  ==================================================
  ```

- **Debug Mode**: Shows logging output and adds configuration information to each result
  ```
  Using API key from environment variable: OPENAI_API_KEY
  2025-03-15 15:22:14,331 - INFO - OpenAI provider initialized
  ...
  == RESULT: intro ==
  Configuration: provider=openai, model=gpt-4o, temperature=0.5...
  [Output text...]
  ```
- **Verbose Mode**: Shows complete configuration, original prompt text, and detailed logging (including module names)

## Code Style Guidelines
- **Python Version**: 3.6+ compatible code
- **Imports**: Group standard library, third-party, and local imports with blank lines between
- **Typing**: Use type hints for function parameters and return values
- **Error Handling**: Use try/except blocks with specific exception handling
- **Logging**: Use the logging module with appropriate log levels (not print statements)
- **Class Structure**: Follow ABC pattern for providers with common interface
- **Docstrings**: Use Google-style docstrings for classes and methods
- **Variable Naming**: Use snake_case for variables and functions, CamelCase for classes
- **Formatting**: 4-space indentation, max line length ~100 characters
- **Parameter Handling**: Use **kwargs for flexible parameter passing with defaults
- **API Design**: Maintain backward compatibility when adding new features

# Using the LLM Inference Framework with Claude

This document provides guidance on using the LLM Inference Framework with the Anthropic Claude model.

## Supported Claude Models
The framework currently supports the following Claude models:

- `claude-3-sonnet-20240229`

You can list the available Claude models using the following command:

```bash
python cli.py --provider anthropic --list-models
```

## Example Usage

To use the Claude model, you can run the following command:

```bash
python cli.py -p anthropic -m claude-3-sonnet-20240229 "Explain quantum computing in simple terms."
```

This will use the `claude-3-sonnet-20240229` model to generate a response for the given prompt.

You can also use the framework's YAML configuration to run prompts with the Claude model:

```yaml
prompts:
  - id: "claude_example"
    provider: "anthropic"
    model: "claude-3-sonnet-20240229"
    prompt: "Explain machine learning concepts"
    temperature: 0.7
    max_tokens: 500
```

Save this YAML file (e.g., `example-prompts.yaml`) and run:

```bash
python cli.py example-prompts.yaml --provider anthropic
```

This will execute the prompt defined in the YAML file using the Claude model.