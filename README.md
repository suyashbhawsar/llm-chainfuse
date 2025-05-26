# LLM ChainFuse - Multi-Provider LLM Inference Framework

## 🚀 Overview
A **powerful and flexible** framework for working with Large Language Models across multiple providers. Build complex AI workflows with ease!

**Key Capabilities:**
- 🌐 **Multi-Provider Support** – OpenAI, Anthropic Claude, Google Gemini, and Ollama
- 🔗 **Prompt Chaining** – Use outputs from one prompt as input for another
- ⚡ **Parallel Processing** – Run independent prompts simultaneously
- 🎯 **Fine-Grained Control** – Customize temperature, tokens, models per prompt
- 📊 **Real-time Streaming** – Get responses as they're generated
- 🔍 **Debug & Validation** – Comprehensive debugging and model validation
- 🖥️ **CLI & Python API** – Use as command-line tool or Python library

---

## 🛠️ Quick Start

### **1. Install Dependencies**
```bash
uv sync
source .venv/bin/activate
```

### **2. Set API Keys**
```bash
# Option 1: Environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"

# Option 2: Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### **3. Run Your First Workflow**
```bash
# Basic usage
python main.py example-prompts.yaml

# With streaming
python main.py -p openai -m gpt-4o --stream "Tell me about AI"

# List available models
python main.py --provider openai --list-models
```

---

## 📝 Creating Workflows

### **Simple Prompt Chain**
Create a YAML file with your prompts:

```yaml
prompts:
  - id: "research"
    provider: "openai"
    model: "gpt-4o"
    prompt: "Research the basics of quantum computing"
    temperature: 0.7
    max_tokens: 300

  - id: "summary"
    provider: "anthropic"
    model: "claude-3-sonnet-20240229"
    prompt: "Summarize this research: {{ research }}"
    temperature: 0.5
    max_tokens: 150

  - id: "code_example"
    provider: "openai"
    model: "gpt-4o"
    prompt: "Create Python code based on: {{ summary }}"
    temperature: 0.2
    max_tokens: 400
```

### **Multi-Provider Workflow**
```yaml
prompts:
  # OpenAI for creative writing
  - id: "story_idea"
    provider: "openai"
    model: "gpt-4o"
    prompt: "Create a sci-fi story concept"
    temperature: 0.9

  # Claude for analysis
  - id: "story_analysis"
    provider: "anthropic"
    model: "claude-3-sonnet-20240229"
    prompt: "Analyze this story concept: {{ story_idea }}"
    temperature: 0.3

  # Gemini for expansion
  - id: "story_expansion"
    provider: "gemini"
    model: "gemini-1.5-flash"
    prompt: "Expand on this analysis: {{ story_analysis }}"
    temperature: 0.7

  # Local Ollama for code generation
  - id: "story_code"
    provider: "ollama"
    model: "llama3.2"
    prompt: "Write Python code to generate stories like: {{ story_idea }}"
    temperature: 0.1
```

---

## 🖥️ CLI Usage Examples

### **Basic Commands**
```bash
# Run workflow
python main.py prompts.yaml

# Save results to file
python main.py prompts.yaml --output results.json

# Run specific prompt only
python main.py prompts.yaml --run-id summary

# Override global settings
python main.py prompts.yaml --temperature 0.8 --model gpt-4o
```

### **Streaming & Real-time**
```bash
# Stream responses in real-time
python main.py -p openai --stream "Explain machine learning"

# Stream with different providers
python main.py -p anthropic --stream "What is quantum computing?"
python main.py -p ollama -m llama3.2 --stream "Write a Python function"
```

### **Debug & Validation**
```bash
# Debug mode with timing
python main.py prompts.yaml --debug

# Validate models before running
python main.py prompts.yaml --validate-models

# Show detailed output
python main.py prompts.yaml --print summary analysis --show-prompts
```

### **Provider Management**
```bash
# List all providers
python main.py --list-providers

# List models for specific provider
python main.py --provider openai --list-models
python main.py --provider anthropic --list-models
python main.py --provider gemini --list-models
python main.py --provider ollama --list-models
```

---

## 🐍 Python API

### **Basic Usage**
```python
from llm_inference import LLMInference

# Initialize with provider
llm = LLMInference(provider="openai")

# Single prompt
response = llm.call_api(
    "Explain DevOps", 
    model="gpt-4o",
    temperature=0.7
)

# Run workflow from file
results = llm.run("workflow.yaml", output_file="results.json")
```

### **Advanced Usage**
```python
# Multi-provider setup
providers = {
    "creative": LLMInference(provider="openai"),
    "analytical": LLMInference(provider="anthropic"), 
    "local": LLMInference(provider="ollama")
}

# Custom workflow
creative_response = providers["creative"].call_api(
    "Write a creative story", 
    model="gpt-4o", 
    temperature=0.9
)

analysis = providers["analytical"].call_api(
    f"Analyze this story: {creative_response}",
    model="claude-3-sonnet-20240229",
    temperature=0.3
)

# Validate before running
llm.validate_models("complex-workflow.yaml")
```

---

## 🔧 Configuration Options

### **Prompt Parameters**
Each prompt supports these parameters:

| Parameter | OpenAI | Anthropic | Gemini | Ollama | Description |
|-----------|--------|-----------|--------|---------|-------------|
| `model` | ✅ | ✅ | ✅ | ✅ | Model to use |
| `temperature` | ✅ | ✅ | ✅ | ✅ | Creativity (0-1) |
| `max_tokens` | ✅ | ✅ | ✅ | ✅ | Response length |
| `top_p` | ✅ | ✅ | ✅ | ✅ | Nucleus sampling |
| `seed` | ✅ | ❌ | ❌ | ❌ | Reproducibility |
| `frequency_penalty` | ✅ | ❌ | ❌ | ❌ | Reduce repetition |
| `presence_penalty` | ✅ | ❌ | ❌ | ❌ | Encourage diversity |
| `top_k` | ❌ | ✅ | ✅ | ❌ | Top-k sampling |

### **Output Control**
```yaml
# Control what gets printed
print:
  print_all: false
  print_ids: ["summary", "final_result"]

prompts:
  # Your prompts here...
```

### **Context Files**
Include external files as context:
```bash
# Include text files
python main.py prompts.yaml --context background:context.txt

# Include YAML data with dot notation access
python main.py prompts.yaml --context data:products.yaml
```

---

## 🌟 Advanced Features

### **Parallel Processing**
Independent prompts run automatically in parallel:
```yaml
prompts:
  # These run simultaneously
  - id: "task_a"
    prompt: "Research topic A"
  - id: "task_b" 
    prompt: "Research topic B"
  
  # This waits for both above to complete
  - id: "combined"
    prompt: "Combine insights: {{ task_a }} and {{ task_b }}"
```

### **Dependency Resolution**
The framework automatically detects dependencies and optimizes execution order.

### **Error Handling**
Robust error handling with detailed logging and graceful degradation.

### **Model Validation**
Validates model availability and parameter compatibility before execution.

---

## 🎯 Use Cases

### **Content Creation Pipeline**
```yaml
prompts:
  - id: "outline"
    prompt: "Create an outline for: {{ topic }}"
    
  - id: "draft"
    prompt: "Write content based on: {{ outline }}"
    
  - id: "edit"
    provider: "anthropic"
    prompt: "Edit and improve: {{ draft }}"
    
  - id: "seo"
    prompt: "Add SEO optimization to: {{ edit }}"
```

### **Code Development Workflow**
```yaml
prompts:
  - id: "requirements"
    prompt: "Analyze requirements for: {{ project_description }}"
    
  - id: "architecture"
    prompt: "Design architecture based on: {{ requirements }}"
    
  - id: "code"
    provider: "openai"
    model: "gpt-4o"
    prompt: "Generate code for: {{ architecture }}"
    
  - id: "tests"
    prompt: "Create tests for: {{ code }}"
    
  - id: "documentation"
    provider: "anthropic"
    prompt: "Document this code: {{ code }}"
```

### **Research & Analysis**
```yaml
prompts:
  - id: "research"
    prompt: "Research the latest developments in: {{ topic }}"
    
  - id: "analysis"
    provider: "anthropic"
    prompt: "Analyze trends in: {{ research }}"
    
  - id: "insights"
    prompt: "Extract key insights from: {{ analysis }}"
    
  - id: "recommendations"
    prompt: "Provide actionable recommendations based on: {{ insights }}"
```

---

## 🔐 Environment Setup

The framework supports multiple ways to securely manage API keys:

### **Basic .env Setup**
```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-gemini-key
```

### **Secure Credential Retrieval**
For enhanced security, use executable commands:
```env
# 1Password CLI
OPENAI_API_KEY=exec:op read "op://vault/openai/credential"

# macOS Keychain
ANTHROPIC_API_KEY=exec:security find-internet-password -s anthropic.com -w

# Environment variables
GOOGLE_API_KEY=exec:echo $GOOGLE_TOKEN
```

---

## 🛠️ Development & Contributing

### **Adding New Providers**
1. Inherit from `ModelProvider` in `model_providers.py`
2. Implement required methods: `initialize`, `list_models`, `generate`, etc.
3. Add to the `get_provider` factory function

### **Project Structure**
```
llm-chainfuse/
├── main.py              # CLI entry point
├── llm_inference.py     # Core inference engine
├── model_providers.py   # Provider implementations
├── env_loader.py        # Environment management
├── example-prompts.yaml # Example workflows
└── README.md           # This file
```

---

## 📊 Output Examples

### **Standard Output**
```
=== LLM INFERENCE STATUS ===
✅ Prompt 'research': Success (2.1s)
✅ Prompt 'summary': Success (1.8s)
✅ Prompt 'analysis': Success (3.2s)
```

### **Debug Mode**
```
=== Dependency Graph ===
analysis -> [research, summary]

=== Parallel Execution ===
Running: research, summary (2 prompts)
Waiting for dependencies: analysis

=== Results ===
research: "Quantum computing represents..."
summary: "Key points include..."
analysis: "Based on the research and summary..."
```

---

## 🚀 What's Next?

- [x] Multi-provider support ✅
- [x] Parallel processing ✅  
- [x] Streaming responses ✅
- [x] Enhanced environment management ✅
- [ ] Token usage tracking
- [ ] Response caching
- [ ] Advanced retry mechanisms
- [ ] Web UI interface
- [ ] Plugin system for custom providers

---

**Ready to build powerful LLM workflows?** 🚀 Start with the examples above and customize for your needs!
