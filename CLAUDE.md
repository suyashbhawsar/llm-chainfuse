# CLAUDE.md: Repository Guidelines

## Build & Run Commands
- **Core Installation**: `pip install -r requirements.txt`
- **Basic Run**: `python cli.py example-prompts.yaml` (prints all results to terminal)
- **Save Output to File**: `python cli.py example-prompts.yaml -o results.json` (prints results and saves to file)
- **Hide Terminal Output**: `python cli.py example-prompts.yaml -o results.json --silent` (only saves to file)
- **Print Specific Results**: `python cli.py example-prompts.yaml --print summary critique` (only prints specified prompts)
- **Show Prompts**: `python cli.py example-prompts.yaml --show-prompts` (shows original and resolved prompts with results)
- **Debug Mode**: `python cli.py example-prompts.yaml --debug` (shows configs and more info about LLM results)
- **Verbose Debug**: `python cli.py example-prompts.yaml --debug -v` (displays full prompts and detailed configs)
- **Model Listing**: `python cli.py --provider openai --list-models`
- **Validate Models**: `python cli.py example-prompts.yaml --validate-models`
- **Use with Context**: `python cli.py example-prompts.yaml --context background:context1.txt examples:context2.txt`
- **Direct Prompt**: `python cli.py "Explain quantum computing in simple terms"`

## Output Format
- **Standard Run**: Shows complete results for each prompt
  ```
  === LLM INFERENCE RESULTS ===

  == RESULT: intro ==

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