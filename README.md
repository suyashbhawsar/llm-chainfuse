# LLM Inference Framework

## 🚀 Overview
This project provides a **flexible and developer-friendly** framework to interact with Large Language Models (LLMs). It supports:

- **Multiple LLM Providers** 🌐 – Work with OpenAI, Anthropic Claude, Ollama, and Google Gemini.
- **Independent Prompts** 🚀 – Process multiple prompts in parallel.
- **Chained Prompts (Piping)** 🔗 – Use outputs of one prompt as input for another.
- **Advanced Parameter Tuning** 🎯 – Customize model, temperature, top_p, max_tokens, seed, penalties, etc.
- **Reproducibility with Seed** 🔄 – Ensure deterministic outputs.
- **CLI and Python API Support** 🖥️ – Use as a command-line tool or import as a library.
- **Model Validation & Listing** 📜 – Fetch available models and verify parameter compatibility.
- **Token Streaming** ⚡ – Get responses as they are generated, token by token.
- **Debug Mode** 🔍 – Comprehensive debugging with timing information and prompt visibility.

---

## 📌 Features

| Feature               | Description |
|----------------------|-------------|
| **Multi-Provider Support** | Work with OpenAI, Anthropic Claude, and Ollama through a unified interface. |
| **Model Selection** | Specify any supported model per prompt, with the right provider handling. |
| **Temperature Control** | Set temperature for creativity control. |
| **Top-P Sampling** | Alternative sampling method to temperature. |
| **Token Limits** | Customize `max_tokens` per prompt. |
| **Frequency & Presence Penalty** | Adjust response diversity and repetition (OpenAI only). |
| **Seed for Reproducibility** | Ensure consistent responses with the same input (OpenAI only). |
| **Piped Prompts** | Use `{{ variable_name }}` syntax to reuse earlier outputs. |
| **Parallel Execution** | Process independent prompts simultaneously with ThreadPoolExecutor. |
| **Dependency Management** | Automatically detect and resolve prompt dependencies. |
| **Debug Mode** | Comprehensive debugging with timing information and prompt visibility. |
| **CLI and Module Support** | Use as a CLI tool or import as a Python library. |
| **Mixed Provider Workflows** | Chain prompts across different providers in the same workflow. |
| **Context Files** | Include external text files as context in your prompts. |
| **Model Listing & Validation** | Fetch available models and verify parameter compatibility. |
| **Token Streaming** | Get responses as they are generated, token by token, for real-time output. |

---

## 📂 Installation

### **1️⃣ Install Dependencies**
```bash
uv sync
```

### **2️⃣ Set API Keys**
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-gemini-api-key"

# For Ollama - no API key needed (runs locally)
```
(For Windows, use `set OPENAI_API_KEY=your-api-key-here`)

---

## 🖥️ CLI Usage

### **Run Inference** (Basic Example)
```bash
# Using YAML configuration (results shown immediately as they arrive)
python main.py example-prompts.yaml

# Save output to file while showing results
python main.py example-prompts.yaml --output results.json

# Wait for all results before displaying (batch mode)
python main.py example-prompts.yaml --batch

# Using JSON configuration
python main.py example-prompts.json --output results.json
```

### **Run with Specific Provider**
```bash
python main.py example-prompts.yaml --provider anthropic
python main.py example-prompts.yaml --provider gemini
```

### **Override Defaults from CLI**
```bash
python main.py example-prompts.yaml --temperature 0.9 --max_tokens 500 --model gpt-4o --output results.json
```

### **Streaming Support**
```bash
# Stream direct prompt response (streaming only works with direct prompts)
python main.py -p openai -m gpt-4o --stream "Tell me a story"

# Stream with specific provider
python main.py -p anthropic -m claude-3-sonnet-20240229 --stream "Explain quantum computing"
python main.py -p gemini -m gemini-1.5-flash --stream "What is machine learning?"

# Stream with Ollama
python main.py -p ollama -m llama3.2:latest --stream "What is machine learning?"
```

### **List Available Models**
```bash
python main.py --provider openai --list-models
python main.py --provider anthropic --list-models
python main.py --provider gemini --list-models
python main.py --provider ollama --list-models
```

### **List Available Providers**
```bash
python main.py --list-providers
```

### **Validate Model & Parameters Before Execution**
```bash
python main.py example-prompts.yaml --validate-models
```

### **Print Results to Console**
```bash
# Results are shown immediately as they arrive (default behavior)
python main.py example-prompts.yaml

# Wait for all results before displaying
python main.py example-prompts.yaml --batch

# Print specific prompt results
python main.py example-prompts.yaml --print summary critique

# Hide terminal output (only save to file)
python main.py example-prompts.yaml --output results.json --silent
```

### **Debug Mode Options**
```bash
# Enable debug mode (shows prompts and timing information)
python main.py example-prompts.yaml --debug

# Increase verbosity level (0-2)
python main.py example-prompts.yaml --debug --verbose 2

# Show prompts in output (enabled by default in debug mode)
python main.py example-prompts.yaml --show-prompts

# Show configuration in output
python main.py example-prompts.yaml --show-config

# Show request options in output
python main.py example-prompts.yaml --show-request-options

# Show response headers in output
python main.py example-prompts.yaml --show-response-headers

# Show request ID in output
python main.py example-prompts.yaml --show-request-id
```

### **Parallel Execution Example**
```yaml
prompts:
  # Independent prompts (processed in parallel)
  person_name:
    text: "My name is John."
    model: "gpt-4o"
    max_tokens: 50
    provider: "openai"

  person_location:
    text: "I live in New York."
    model: "gpt-4o"
    max_tokens: 50
    provider: "openai"

  # Dependent prompts (processed after dependencies)
  personal_summary:
    text: "Create a summary about {{ person_name }} who lives in {{ person_location }}"
    model: "gpt-4o"
    max_tokens: 150
    provider: "openai"
```

### **Debug Mode Output Example**
```
=== Prompt Processing Times ===
person_name: 1.37 seconds
person_location: 1.14 seconds
personal_summary: 2.61 seconds

=== Dependency Graph ===
{
    'personal_summary': ['person_name', 'person_location']
}

=== Independent Prompts ===
['person_name', 'person_location']

=== Dependent Prompts ===
['personal_summary']
```

### **Include Context Files**
```bash
python main.py example-prompts.yaml --context background:context1.txt examples:context2.txt
```

You can also include YAML files as context, which supports nested path resolution using dot notation:

```bash
python main.py example-prompts.yaml --context catalog:data/product_catalog.yaml
```

Example YAML context file (`data/product_catalog.yaml`):
```yaml
products:
  electronics:
    laptops:
      - name: "MacBook Pro"
        price: 1299.99
        specs:
          processor: "M2"
          memory: "16GB"
      - name: "ThinkPad X1"
        price: 1499.99
        specs:
          processor: "Intel i7"
          memory: "32GB"
    phones:
      - name: "iPhone 15"
        price: 999.99
        specs:
          storage: "256GB"
          color: "Space Gray"
  accessories:
    headphones:
      - name: "AirPods Pro"
        price: 249.99
        specs:
          type: "Wireless"
          battery: "24h"
```

In your prompts, you can access nested values using dot notation:
```yaml
prompts:
  - id: "product_description"
    prompt: "Create a marketing description for the {{ catalog.products.electronics.laptops.0.name }} with {{ catalog.products.electronics.laptops.0.specs.processor }} processor."
```

The system will automatically resolve the nested paths and replace the placeholders with the corresponding values.

---

## 📝 Prompt Configuration

Prompts can be defined in either **YAML** or **JSON** format with **custom parameters per prompt**.

### YAML Example:

```yaml
# Optional printing configuration
print:
  print_all: false    # Set to true to print all results in YAML/JSON config
  print_ids:          # List of specific prompt IDs to print
    - summary
    - critique
  # Note: CLI args (--print, --debug, -v) will override these settings

prompts:
  - id: "intro"
    provider: "openai"    # Optional, defaults to CLI provider or "openai"
    prompt: "Explain DevOps in simple terms."
    model: "gpt-4o"
    temperature: 0.5
    top_p: 0.9
    seed: 42
    max_tokens: 300
    frequency_penalty: 0.1
    presence_penalty: 0.1

  - id: "summary"
    provider: "anthropic"
    model: "claude-3-sonnet-20240229"
    prompt: "Summarize the following: {{ intro }}"
    temperature: 0.7
    max_tokens: 150

  - id: "code_example"
    provider: "ollama"
    model: "llama3.2"
    prompt: "Create Python code based on this summary: {{ summary }}"
    temperature: 0.2
    max_tokens: 500

  - id: "gemini_example"
    provider: "gemini"
    model: "gemini-1.5-flash"
    prompt: "Explain machine learning concepts"
    temperature: 0.7
    max_tokens: 500

  - id: "independent"
    prompt: "What are the key principles of Site Reliability Engineering?"
    model: "gpt-4-turbo"
    temperature: 0.6
    seed: 1234
```

### JSON Example:

```json
{
  "print": {
    "print_all": false,
    "print_ids": ["summary", "critique"],
    "_comment": "CLI args (--print, --debug, -v) will override these settings"
  },
  "prompts": [
    {
      "id": "intro",
      "provider": "openai",
      "prompt": "Explain DevOps in simple terms.",
      "model": "gpt-4o",
      "temperature": 0.5,
      "top_p": 0.9,
      "seed": 42,
      "max_tokens": 300,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.1
    },
    {
      "id": "summary",
      "provider": "anthropic",
      "model": "claude-3-sonnet-20240229",
      "prompt": "Summarize the following: {{ intro }}",
      "temperature": 0.7,
      "max_tokens": 150
    },
    {
      "id": "code_example",
      "provider": "ollama",
      "model": "llama3.2",
      "prompt": "Create Python code based on this summary: {{ summary }}",
      "temperature": 0.2,
      "max_tokens": 500
    },
    {
      "id": "gemini_example",
      "provider": "gemini",
      "model": "gemini-1.5-flash",
      "prompt": "Explain machine learning concepts",
      "temperature": 0.7,
      "max_tokens": 500
    },
    {
      "id": "independent",
      "prompt": "What are the key principles of Site Reliability Engineering?",
      "model": "gpt-4-turbo",
      "temperature": 0.6,
      "seed": 1234
    }
  ]
}
```

### **How It Works**
- Each **prompt** runs independently unless it references a previous output using `{{ variable_name }}`.
- If **parameters (temperature, model, etc.) are specified**, they override the defaults.
- **Independent prompts run in parallel** for efficiency.
- **Dependent prompts run sequentially**, replacing `{{ variable_name }}` placeholders with previous outputs.
- **Model and Parameter Validation** ensures that the selected model supports the requested parameters.
- **Different providers** can be used for different prompts in the same workflow.
- **Token Streaming** provides real-time output as the model generates responses (direct prompts only).
- **Output Format Control**:
  - Default mode shows results immediately as they arrive from LLMs
  - Streaming mode shows response tokens as they are generated (direct prompts only)
  - Batch mode waits for all prompts to complete before displaying results
  - Print mode displays complete results with optional prompt display
  - Debug mode reveals configuration details and logs
  - Silent mode suppresses terminal output while still saving to file

---

## 🖥️ Development (Python API)

### **Import as a Library**
```python
from llm_inference import LLMInference

# Default (OpenAI)
llm = LLMInference()
response = llm.call_api("What is CI/CD?", temperature=0.7, model="gpt-4o")
print(response)

# With Anthropic
llm_anthropic = LLMInference(provider="anthropic")
response = llm_anthropic.call_api("Explain ML models", model="claude-3-sonnet-20240229")

# With Ollama
llm_ollama = LLMInference(provider="ollama")
response = llm_ollama.call_api("Write a Python function", model="llama3.2")
```

### **Run with Configuration Files**
```python
# Using YAML
llm.run("example-prompts.yaml", output_file="results.json")

# Using JSON
llm.run("example-prompts.json", output_file="results.json")

# With debug args (pass CLI args to control output format)
import argparse
args = argparse.Namespace(debug=True, verbose=False, print=["intro", "summary"])
llm.run("example-prompts.yaml", output_file="results.json", cli_args=args)
```

### **Validate Model & Parameters in Python**
```python
llm.validate_models("example-prompts.yaml")
```

### **List Available Models in Python**
```python
models = llm.list_available_models()
print(models)
```

---

## 📌 Output Formats

### **Standard Run** (No output flags)
```
=== LLM INFERENCE STATUS ===

✅ Prompt 'intro': Success
✅ Prompt 'summary': Success
✅ Prompt 'code_example': Success
✅ Prompt 'gemini_example': Success
✅ Prompt 'independent': Success

Use --debug flag for more details or --print to see results
```

### **Print Mode** (with `--print`)
```
=== LLM INFERENCE RESULTS ===

== RESULT: intro ==

DevOps is a combination of practices, tools, and cultural philosophies...

==================================================

== RESULT: summary ==

DevOps is an approach that combines practices, tools, and cultural philosophies...

==================================================
```

### **Debug Mode** (with `--print --debug`)
```
Using API key from environment variable: OPENAI_API_KEY
2025-03-15 15:22:14,331 - INFO - OpenAI provider initialized
...

=== LLM INFERENCE RESULTS ===

== RESULT: intro ==
Configuration: provider=openai, model=gpt-4o, temperature=0.5...

DevOps is a combination of practices...
```

### **JSON Output** (`results.json`)
```json
{
    "intro": "DevOps integrates software development and IT operations...",
    "summary": "DevOps improves CI/CD, automation, and collaboration...",
    "code_example": "def deploy_app():\n    # Implementation of CI/CD pipeline\n    ...",
    "gemini_example": "Machine learning is a subset of artificial intelligence...",
    "independent": "SRE emphasizes reliability, automation, monitoring, and error budgets."
}
```

---

## 🔥 Advanced Features

### **1️⃣ YAML and JSON Support**
The framework supports both YAML and JSON formats for prompt configuration:

```bash
# Run with YAML configuration
python main.py examples/input.yaml

# Run with JSON configuration
python main.py examples/input.json
```

### **2️⃣ Provider-Specific Parameters**
Each provider supports different parameters. The framework automatically handles parameter compatibility:

| Parameter | OpenAI | Anthropic | Ollama | Gemini |
|-----------|--------|-----------|--------|---------|
| temperature | ✅ | ✅ | ✅ | ✅ |
| max_tokens | ✅ | ✅ | ✅ (as num_predict) | ✅ |
| top_p | ✅ | ✅ | ✅ | ✅ |
| seed | ✅ | ❌ | ❌ | ❌ |
| frequency_penalty | ✅ | ❌ | ❌ | ❌ |
| presence_penalty | ✅ | ❌ | ❌ | ❌ |
| top_k | ❌ | ✅ | ❌ | ✅ |

### **3️⃣ Override Defaults (Per Prompt or CLI)**
- If a **parameter is defined in YAML/JSON**, it overrides global defaults.
- If **not defined**, it falls back to CLI defaults.

### **4️⃣ Print Configuration**
Specify which prompt results to print directly in your configuration file:

```yaml
# In YAML
print:
  print_all: true  # Print all results
  # OR
  print_ids:       # Print specific prompt IDs
    - summary
    - critique
```

```json
// In JSON
{
  "print": {
    "print_all": true,
    "print_ids": ["summary", "critique"]
  }
}
```

### **5️⃣ Streaming Support**
- Enable real-time streaming output from API responses (direct prompts only).

### **6️⃣ Token Usage Logging** (Future Enhancement)
- Track API token consumption per request.

---

## 🧩 Adding New Providers

To add a new provider:

1. Create a new class that inherits from `ModelProvider` in `model_providers.py`
2. Implement all required methods:
   - `initialize`: Set up the provider client
   - `list_models`: Return available models
   - `generate`: Generate text from a prompt
   - `get_default_params`: Return default parameters
   - `validate_model`: Validate model and parameter compatibility
3. Add your provider to the `get_provider` factory function

---

## 🤝 Contributing

1. **Fork** the repo & create a feature branch.
2. Submit a **pull request** with a clear description.
3. Follow **PEP-8** coding standards.

---

## 📜 License

This project is licensed under the **License**.

---

## 🛠️ Troubleshooting & FAQ

### **Issue: API Key Not Set**
**Solution:** Set the appropriate environment variable for your provider:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-gemini-key"
```

### **Issue: API Call Limit Exceeded**
**Solution:** Reduce `max_tokens` or check your API provider usage limits.

### **Issue: Ollama Connection Failed**
**Solution:** Ensure Ollama is running locally:
```bash
# Start Ollama if not running
ollama run llama3.2
```

### **Issue: CLI Error – Duplicate `prompt` Argument**
**Solution:** Ensure `process_prompts()` correctly filters out `prompt` before passing `**kwargs`.

### **Issue: Provider-Specific Parameters**
**Solution:** Check the parameter compatibility table and only use supported parameters for each provider.

---

## 🎯 Next Steps
- [ ] **Add streaming support for real-time responses**
- [ ] **Add logging for token usage**
- [ ] **Support more LLM providers (Mistral, Google, etc.)**
- [ ] **Enhance error handling & automatic retries**
- [ ] **Add caching mechanism for repeated queries**

🚀 **Enjoy building with LLMs!** Let me know if you need enhancements! 🎯
