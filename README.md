# LLM Inference Framework

## 🚀 Overview
This project provides a **flexible and developer-friendly** framework to interact with Large Language Models (LLMs). It supports:

- **Multiple LLM Providers** 🌐 – Work with OpenAI, Anthropic Claude, and Ollama.
- **Independent Prompts** 🚀 – Process multiple prompts in parallel.
- **Chained Prompts (Piping)** 🔗 – Use outputs of one prompt as input for another.
- **Advanced Parameter Tuning** 🎯 – Customize model, temperature, top_p, max_tokens, seed, penalties, etc.
- **Reproducibility with Seed** 🔄 – Ensure deterministic outputs.
- **CLI and Python API Support** 🖥️ – Use as a command-line tool or import as a library.
- **Model Validation & Listing** 📜 – Fetch available models and verify parameter compatibility.

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
| **Parallel Execution** | Process independent prompts simultaneously. |
| **CLI and Module Support** | Use as a CLI tool or import as a Python library. |
| **Mixed Provider Workflows** | Chain prompts across different providers in the same workflow. |
| **Context Files** | Include external text files as context in your prompts. |
| **Model Listing & Validation** | Fetch available models and verify parameter compatibility. |

---

## 📂 Installation

### **1️⃣ Install Core Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Set API Keys**
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Ollama - no API key needed (runs locally)
```
(For Windows, use `set OPENAI_API_KEY=your-api-key-here`)

---

## 🖥️ CLI Usage

### **Run Inference** (Basic Example)
```bash
# Using YAML configuration
python cli.py example-prompts.yaml --output results.json

# Using JSON configuration
python cli.py example-prompts.json --output results.json
```

### **Run with Specific Provider**
```bash
python cli.py example-prompts.yaml --provider anthropic
```

### **Override Defaults from CLI**
```bash
python cli.py example-prompts.yaml --temperature 0.9 --max_tokens 500 --model gpt-4o --output results.json
```

### **List Available Models**
```bash
python cli.py --provider openai --list-models
python cli.py --provider anthropic --list-models
python cli.py --provider ollama --list-models
```

### **List Available Providers**
```bash
python cli.py --list-providers
```

### **Validate Model & Parameters Before Execution**
```bash
python cli.py example-prompts.yaml --validate-models
```

### **Print Results to Console**
```bash
# Print all results (clean output format)
python cli.py example-prompts.yaml --print

# Print specific prompt results
python cli.py example-prompts.yaml --print summary critique
```

### **Debug Mode**
```bash
# Show detailed debug information including configuration and logs
python cli.py example-prompts.yaml --print --debug

# Enable more verbose debug output with full prompt text
python cli.py example-prompts.yaml --print --debug -v
```

### **Include Context Files**
```bash
python cli.py example-prompts.yaml --context background:context1.txt examples:context2.txt
```

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
- **Output Format Control**:
  - Standard mode shows simple success/failure status
  - Print mode displays complete results
  - Debug mode reveals configuration details and logs
  - All output is displayed in the same order as prompts appear in the configuration file

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
    "independent": "SRE emphasizes reliability, automation, monitoring, and error budgets."
}
```

---

## 🔥 Advanced Features

### **1️⃣ YAML and JSON Support**
The framework supports both YAML and JSON formats for prompt configuration:

```bash
# Run with YAML configuration
python cli.py examples/input.yaml

# Run with JSON configuration
python cli.py examples/input.json
```

### **2️⃣ Provider-Specific Parameters**
Each provider supports different parameters. The framework automatically handles parameter compatibility:

| Parameter | OpenAI | Anthropic | Ollama |
|-----------|--------|-----------|--------|
| temperature | ✅ | ✅ | ✅ |
| max_tokens | ✅ | ✅ | ✅ (as num_predict) |
| top_p | ✅ | ✅ | ✅ |
| seed | ✅ | ❌ | ❌ |
| frequency_penalty | ✅ | ❌ | ❌ |
| presence_penalty | ✅ | ❌ | ❌ |

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

### **5️⃣ Streaming Support** (Coming Soon)
- Enable real-time streaming output from API responses.

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
