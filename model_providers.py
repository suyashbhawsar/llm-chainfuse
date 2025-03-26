import os
import abc
import json
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelProvider(abc.ABC):
    """Abstract base class for LLM model providers"""

    @abc.abstractmethod
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the model provider with API key and other provider-specific parameters"""
        pass

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider"""
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, model: str, **params) -> str:
        """Generate text from the model given a prompt and parameters"""
        pass

    @abc.abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for this model provider"""
        pass

    @abc.abstractmethod
    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate if a model name and parameters are compatible with this provider"""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI-specific model provider implementation"""

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the OpenAI client"""
        try:
            import openai

            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key is missing. Set OPENAI_API_KEY as an environment variable or pass it explicitly.")

            # Handle different versions of OpenAI SDK
            try:
                # New SDK style (>=1.0.0)
                self.client = openai.OpenAI(api_key=self.api_key)
            except AttributeError:
                # Legacy SDK style
                openai.api_key = self.api_key
                self.client = openai

            logging.info("OpenAI provider initialized")
        except ImportError:
            logging.error("OpenAI package not installed. Install with: pip install openai>=1.0.0")
            raise

    def list_models(self) -> List[str]:
        """List available models from OpenAI"""
        try:
            # Check if using new SDK or legacy SDK
            if hasattr(self.client, 'models') and hasattr(self.client.models, 'list'):
                # New SDK style
                models = self.client.models.list()
                return [model.id for model in models.data]
            else:
                # Legacy SDK style
                models = self.client.Model.list()
                return [model.id for model in models.data]
        except Exception as e:
            logging.error(f"Failed to retrieve OpenAI model list: {e}")
            # Return default list of models if API fails
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]

    def generate(self, prompt: str, model: str, **params) -> str:
        """Generate text using OpenAI models"""
        try:
            # Check if using new SDK or legacy SDK
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New SDK style
                # Start with minimal parameters
                api_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                # Only add parameters that are explicitly provided
                if "temperature" in params:
                    api_params["temperature"] = params["temperature"]
                if "max_tokens" in params:
                    api_params["max_tokens"] = params["max_tokens"]
                if "top_p" in params:
                    api_params["top_p"] = params["top_p"]
                if "seed" in params:
                    api_params["seed"] = params["seed"]
                if "frequency_penalty" in params:
                    api_params["frequency_penalty"] = params["frequency_penalty"]
                if "presence_penalty" in params:
                    api_params["presence_penalty"] = params["presence_penalty"]
                if "logprobs" in params:
                    api_params["logprobs"] = params["logprobs"]

                response = self.client.chat.completions.create(**api_params)
                return response.choices[0].message.content.strip()
            else:
                # Legacy SDK style
                # Start with minimal parameters
                api_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                # Only add parameters that are explicitly provided
                if "temperature" in params:
                    api_params["temperature"] = params["temperature"]
                if "max_tokens" in params:
                    api_params["max_tokens"] = params["max_tokens"]
                if "top_p" in params:
                    api_params["top_p"] = params["top_p"]
                if "seed" in params:
                    api_params["seed"] = params["seed"]
                if "frequency_penalty" in params:
                    api_params["frequency_penalty"] = params["frequency_penalty"]
                if "presence_penalty" in params:
                    api_params["presence_penalty"] = params["presence_penalty"]
                if "logprobs" in params:
                    api_params["logprobs"] = params["logprobs"]

                response = self.client.ChatCompletion.create(**api_params)
                return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for OpenAI models"""
        # Return empty dictionary as we're only using explicitly provided parameters
        return {}

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate OpenAI model and parameters"""
        available_models = self.list_models()
        issues = []

        if not available_models:
            issues.append("Unable to fetch the latest model list. Skipping validation.")
            return issues

        if model_name not in available_models:
            issues.append(f"Model '{model_name}' is not a valid OpenAI model.")

        # Check if parameters are supported based on model family
        if model_name.startswith("gpt-4") and params.get("seed") is not None:
            issues.append(f"Model '{model_name}' does not support: seed")

        if model_name.startswith("gpt-3") and params.get("top_p") is not None:
            issues.append(f"Model '{model_name}' does not support: top_p")

        return issues


class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider implementation"""

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            import anthropic

            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key is missing. Set ANTHROPIC_API_KEY as an environment variable or pass it explicitly.")

            self.client = anthropic.Anthropic(api_key=self.api_key)
            logging.info("Anthropic provider initialized")
        except ImportError:
            logging.error("Anthropic package not installed. Install with: pip install anthropic")
            raise

    def list_models(self) -> List[str]:
        """List available Anthropic models"""
        # Anthropic doesn't have a list models endpoint, so we hardcode the available models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20240627",
            # Add latest models as they become available
        ]

    def generate(self, prompt: str, model: str, **params) -> str:
        """Generate text using Anthropic Claude models"""
        try:
            # Create minimal parameters
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }

            # Only add parameters that are explicitly provided and supported by Anthropic
            if "temperature" in params:
                api_params["temperature"] = params["temperature"]
            if "max_tokens" in params:
                api_params["max_tokens"] = params["max_tokens"]
            if "top_p" in params:
                api_params["top_p"] = params["top_p"]

            response = self.client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            return None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for Anthropic models"""
        # Return empty dictionary as we're only using explicitly provided parameters
        return {}

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate Anthropic model and parameters"""
        available_models = self.list_models()
        issues = []

        if model_name not in available_models:
            issues.append(f"Model '{model_name}' is not a valid Anthropic model.")

        # Identify unsupported parameters that have non-default values
        unsupported_params = []
        for param in params:
            if param not in ["temperature", "max_tokens", "top_p"] and params.get(param) is not None:
                unsupported_params.append(param)

        if unsupported_params:
            issues.append(f"Claude models do not support: {', '.join(unsupported_params)}")

        return issues


class OllamaProvider(ModelProvider):
    """Ollama local model provider implementation"""

    def __init__(self):
        """Initialize instance variables"""
        self.base_url = "http://localhost:11434"
        self.session = None

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the Ollama client"""
        # Ollama is local and doesn't need an API key
        self.base_url = kwargs.get("base_url", "http://localhost:11434")

        try:
            import requests
            self.session = requests.Session()

            # Test connection to Ollama server
            try:
                response = self.session.get(f"{self.base_url}/api/version", timeout=5)
                if response.status_code == 200:
                    logging.info(f"Ollama provider initialized with base URL: {self.base_url}")
                else:
                    logging.warning(f"Connected to Ollama server but received status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not connect to Ollama server at {self.base_url}: {e}")
                logging.warning("Make sure Ollama is running locally")

        except ImportError:
            logging.error("Requests package not installed. Install with: pip install requests")
            raise

    def list_models(self) -> List[str]:
        """List available models from Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json().get("models", [])]
            else:
                logging.error(f"Failed to retrieve Ollama model list: {response.text}")
                return []
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return []

    def generate(self, prompt: str, model: str, **params) -> str:
        """Generate text using Ollama models"""
        try:
            # Create minimal parameters for both API endpoints
            payload = {
                "model": model,
                "prompt": prompt,
            }

            chat_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

            # Only add parameters that are explicitly provided
            if "temperature" in params:
                payload["temperature"] = params["temperature"]
                chat_payload["temperature"] = params["temperature"]
            if "max_tokens" in params:
                payload["num_predict"] = params["max_tokens"]  # Ollama parameter name is different
                chat_payload["num_predict"] = params["max_tokens"]
            if "top_p" in params:
                payload["top_p"] = params["top_p"]
                chat_payload["top_p"] = params["top_p"]

            try:
                # First try the chat endpoint (newer Ollama versions)
                response = self.session.post(f"{self.base_url}/api/chat", json=chat_payload, timeout=60)
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "").strip()
            except Exception as e:
                logging.warning(f"Ollama chat API failed, falling back to generate API: {e}")

            # Fall back to the generate endpoint (older Ollama versions)
            response = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                # Read the entire response as a string
                response_text = response.text

                # Split the response by newlines to handle Ollama's streaming format
                # Ollama returns each chunk as a separate JSON object, one per line
                lines = response_text.strip().split("\n")

                # Extract and concatenate all response chunks
                full_response = ""
                for line in lines:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                    except json.JSONDecodeError:
                        continue

                return full_response.strip()
            else:
                logging.error(f"Ollama API call failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Ollama API call failed: {e}")
            return None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for Ollama models"""
        # Return empty dictionary as we're only using explicitly provided parameters
        return {}

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate Ollama model and parameters"""
        available_models = self.list_models()
        issues = []

        if not available_models:
            issues.append("Unable to connect to Ollama. Make sure it's running.")
            return issues

        if model_name not in available_models:
            issues.append(f"Model '{model_name}' is not available in Ollama.")

        # Ollama doesn't support these parameters
        unsupported_params = []
        if params.get("seed") is not None:
            unsupported_params.append("seed")
        if params.get("frequency_penalty") is not None:
            unsupported_params.append("frequency_penalty")
        if params.get("presence_penalty") is not None:
            unsupported_params.append("presence_penalty")
        if params.get("logprobs") is not None:
            unsupported_params.append("logprobs")

        if unsupported_params:
            issues.append(f"Ollama models do not support: {', '.join(unsupported_params)}")

        return issues


def get_provider(provider_name: str) -> ModelProvider:
    """Factory function to get the appropriate provider instance"""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        # Add more providers as they are implemented
    }

    if provider_name.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}. Available providers: {', '.join(providers.keys())}")

    return providers[provider_name.lower()]()
