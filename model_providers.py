import os
import abc
import json
import logging
from typing import Dict, List, Optional, Any, Union, Iterator

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
    def generate(self, prompt: str, model: str, stream: bool = False, **params) -> Union[str, Iterator[str]]:
        """Generate text from the model given a prompt and parameters.

        Args:
            prompt: The input prompt
            model: The model to use
            stream: Whether to stream the response
            **params: Additional model parameters

        Returns:
            If stream=False: Complete response as string
            If stream=True: Iterator yielding response chunks
        """
        pass

    @abc.abstractmethod
    def generate_stream(self, **params) -> Iterator[str]:
        """Generate text from the model with streaming enabled.

        Args:
            **params: Model parameters including model name and prompt

        Returns:
            Iterator yielding response chunks
        """
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
    """OpenAI model provider implementation"""

    def __init__(self):
        """Initialize instance variables"""
        self.client = None
        self.api_key = None

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.api_key = api_key
            logging.info("OpenAI provider initialized successfully")
        except ImportError:
            logging.error("OpenAI package not installed. Install with: pip install openai>=1.0.0")
            raise

    def list_models(self) -> List[str]:
        """List available models from OpenAI"""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logging.error(f"Failed to retrieve OpenAI model list: {e}")
            return []

    def generate(self, prompt: str, model: str, stream: bool = False, **params) -> Union[str, Iterator[str]]:
        """Generate text using OpenAI models"""
        try:
            # Check if using new SDK or legacy SDK
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New SDK style
                # Start with minimal parameters
                api_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": stream
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

                if stream:
                    return self.generate_stream(**api_params)
                else:
                    response = self.client.chat.completions.create(**api_params)
                    return response.choices[0].message.content.strip()
            else:
                # Legacy SDK style
                api_params = {
                    "model": model,
                    "prompt": prompt,
                    "stream": stream
                }

                # Add parameters that are explicitly provided
                for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                    if param in params:
                        api_params[param] = params[param]

                if stream:
                    return self.generate_stream(**api_params)
                else:
                    response = self.client.completions.create(**api_params)
                    return response.choices[0].text.strip()

        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return None

    def generate_stream(self, **params) -> Iterator[str]:
        """Generate text using OpenAI models with streaming enabled"""
        try:
            # Check if using new SDK or legacy SDK
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New SDK style
                stream = self.client.chat.completions.create(**params)
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            else:
                # Legacy SDK style
                stream = self.client.completions.create(**params)
                for chunk in stream:
                    if chunk.choices[0].text:
                        yield chunk.choices[0].text

        except Exception as e:
            logging.error(f"OpenAI streaming API call failed: {e}")
            yield None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for OpenAI models"""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate if a model name and parameters are compatible with OpenAI"""
        issues = []
        try:
            # Check if model exists
            response = self.client.models.retrieve(model_name)
            if not response:
                issues.append(f"Model '{model_name}' not found in OpenAI's model list")
        except Exception as e:
            issues.append(f"Error validating model '{model_name}': {e}")

        # Validate parameters
        if "temperature" in params and not 0 <= params["temperature"] <= 2:
            issues.append("Temperature must be between 0 and 2")
        if "max_tokens" in params and params["max_tokens"] < 1:
            issues.append("max_tokens must be positive")
        if "top_p" in params and not 0 <= params["top_p"] <= 1:
            issues.append("top_p must be between 0 and 1")
        if "frequency_penalty" in params and not -2 <= params["frequency_penalty"] <= 2:
            issues.append("frequency_penalty must be between -2 and 2")
        if "presence_penalty" in params and not -2 <= params["presence_penalty"] <= 2:
            issues.append("presence_penalty must be between -2 and 2")

        return issues


class AnthropicProvider(ModelProvider):
    """Anthropic model provider implementation"""

    def __init__(self):
        """Initialize instance variables"""
        self.client = None
        self.api_key = None
        # Anthropic-specific default max_tokens
        self.default_max_tokens = 4096

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.api_key = api_key
            logging.info("Anthropic provider initialized successfully")
        except ImportError:
            logging.error("Anthropic package not installed. Install with: pip install anthropic")
            raise

    def list_models(self) -> List[str]:
        """List available models from Anthropic"""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logging.error(f"Failed to retrieve Anthropic model list: {e}")
            return []

    def generate(self, prompt: str, model: str, stream: bool = False, **params) -> Union[str, Iterator[str]]:
        """Generate text using Anthropic models"""
        try:
            # Start with minimal parameters
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": params.get("max_tokens", self.default_max_tokens),  # Anthropic-specific default
                "stream": stream
            }

            # Only add parameters that are explicitly provided
            if "temperature" in params:
                api_params["temperature"] = params["temperature"]
            if "top_p" in params:
                api_params["top_p"] = params["top_p"]
            if "top_k" in params:
                api_params["top_k"] = params["top_k"]

            if stream:
                return self.generate_stream(**api_params)
            else:
                response = self.client.messages.create(**api_params)
                return response.content[0].text.strip()

        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            return None

    def generate_stream(self, **params) -> Iterator[str]:
        """Generate text using Anthropic models with streaming enabled"""
        try:
            # Ensure required parameters are present (Anthropic-specific)
            if "max_tokens" not in params:
                params["max_tokens"] = self.default_max_tokens
            if "messages" not in params:
                raise ValueError("messages parameter is required for streaming")
            if "model" not in params:
                raise ValueError("model parameter is required for streaming")

            stream = self.client.messages.create(**params)
            for chunk in stream:
                # Handle different event types in Anthropic's streaming API
                if hasattr(chunk, 'type'):
                    if chunk.type == 'content_block_delta':
                        yield chunk.delta.text
                    elif chunk.type == 'message_delta':
                        if hasattr(chunk.delta, 'text'):
                            yield chunk.delta.text
                    elif chunk.type == 'content_block_start':
                        continue
                    elif chunk.type == 'content_block_stop':
                        continue
                    elif chunk.type == 'message_start':
                        continue
                    elif chunk.type == 'message_stop':
                        continue
                # Fallback for older API versions
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    yield chunk.delta.text

        except Exception as e:
            logging.error(f"Anthropic streaming API call failed: {e}")
            yield None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for Anthropic models"""
        return {
            "temperature": 0.7,
            "max_tokens": self.default_max_tokens,  # Anthropic-specific default
            "top_p": 1.0
        }

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate if a model name and parameters are compatible with Anthropic"""
        issues = []
        try:
            # Check if model exists
            response = self.client.models.retrieve(model_name)
            if not response:
                issues.append(f"Model '{model_name}' not found in Anthropic's model list")
        except Exception as e:
            issues.append(f"Error validating model '{model_name}': {e}")

        # Validate parameters
        if "temperature" in params and not 0 <= params["temperature"] <= 1:
            issues.append("Temperature must be between 0 and 1")
        if "max_tokens" in params and params["max_tokens"] < 1:
            issues.append("max_tokens must be positive")
        if "top_p" in params and not 0 <= params["top_p"] <= 1:
            issues.append("top_p must be between 0 and 1")
        if "top_k" in params and params["top_k"] < 1:
            issues.append("top_k must be positive")

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

    def generate(self, prompt: str, model: str, stream: bool = False, **params) -> Union[str, Iterator[str]]:
        """Generate text using Ollama models"""
        try:
            # Create minimal parameters for both API endpoints
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }

            chat_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": stream
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

            if stream:
                return self.generate_stream(**chat_payload)
            else:
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

    def generate_stream(self, **params) -> Iterator[str]:
        """Generate text using Ollama models with streaming enabled"""
        try:
            # Try the chat endpoint first (newer Ollama versions)
            try:
                response = self.session.post(f"{self.base_url}/api/chat", json=params, stream=True, timeout=60)
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    yield chunk["message"]["content"]
                            except json.JSONDecodeError:
                                continue
                    return
            except Exception as e:
                logging.warning(f"Ollama chat API failed, falling back to generate API: {e}")

            # Fall back to the generate endpoint (older Ollama versions)
            response = self.session.post(f"{self.base_url}/api/generate", json=params, stream=True, timeout=60)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue
            else:
                logging.error(f"Ollama streaming API call failed with status {response.status_code}: {response.text}")
                yield None

        except Exception as e:
            logging.error(f"Ollama streaming API call failed: {e}")
            yield None

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for Ollama models"""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate if a model name and parameters are compatible with Ollama"""
        issues = []
        try:
            # Check if model exists
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if model_name not in available_models:
                    issues.append(f"Model '{model_name}' not found in Ollama's model list")
            else:
                issues.append("Failed to retrieve Ollama model list")
        except Exception as e:
            issues.append(f"Error validating model '{model_name}': {e}")

        # Validate parameters
        if "temperature" in params and not 0 <= params["temperature"] <= 1:
            issues.append("Temperature must be between 0 and 1")
        if "max_tokens" in params and params["max_tokens"] < 1:
            issues.append("max_tokens must be positive")
        if "top_p" in params and not 0 <= params["top_p"] <= 1:
            issues.append("top_p must be between 0 and 1")

        return issues


class GeminiProvider(ModelProvider):
    """Google Gemini model provider implementation"""

    def __init__(self):
        """Initialize instance variables"""
        self.client = None
        self.api_key = None
        self.default_model = "gemini-1.5-flash"  # Updated to a widely available model

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the Gemini client"""
        try:
            import google.generativeai as genai
            # Use GOOGLE_API_KEY environment variable if api_key is not provided
            if api_key is None:
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")

            genai.configure(api_key=api_key)
            self.client = genai
            self.api_key = api_key
            logging.info("Gemini provider initialized successfully")
        except ImportError:
            logging.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
            raise

    def list_models(self) -> List[str]:
        """List available models from Gemini"""
        try:
            # Attempt to dynamically get the list of models
            models = self.client.list_models()
            model_names = [model.name.split('/')[-1] for model in models if "generateContent" in model.supported_generation_methods]
            return model_names
        except Exception as e:
            logging.warning(f"Could not retrieve Gemini models dynamically: {e}")
            # Fallback to commonly available models
            return [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
                "gemini-pro-vision"
            ]

    def generate(self, prompt: str, model: str, stream: bool = False, **params) -> Union[str, Iterator[str]]:
        """Generate text using Gemini models"""
        try:
            # Strip 'models/' prefix if it exists
            if model.startswith("models/"):
                model = model.replace("models/", "")

            # Handle legacy model names
            if model == "gemini-pro":
                model = "gemini-1.5-pro"  # Map to the stable successor
                logging.info(f"Model 'gemini-pro' mapped to '{model}'")
            elif model == "gemini-pro-vision":
                # Keep the same name as it appears to be still supported
                logging.info(f"Using model 'gemini-pro-vision'")
            elif model == "gemini-2.0-pro":
                # 2.0-pro isn't available yet, use 1.5-pro instead
                model = "gemini-1.5-pro"
                logging.info(f"Model 'gemini-2.0-pro' not available, using '{model}' instead")
            elif model == "gemini-2.0-flash" and "gemini-2.0-flash" not in self.list_models():
                # If 2.0-flash isn't available, use 1.5-flash
                model = "gemini-1.5-flash"
                logging.info(f"Model 'gemini-2.0-flash' not available, using '{model}' instead")

            # Get the model
            model_instance = self.client.GenerativeModel(model)

            # Start with minimal parameters
            generation_config = {}
            if "temperature" in params:
                generation_config["temperature"] = params["temperature"]
            if "max_tokens" in params:
                generation_config["max_output_tokens"] = params["max_tokens"]
            if "top_p" in params:
                generation_config["top_p"] = params["top_p"]
            if "top_k" in params:
                generation_config["top_k"] = params["top_k"]

            if stream:
                return self.generate_stream(model_instance, prompt, generation_config)
            else:
                # Generate content directly (don't use chat)
                try:
                    response = model_instance.generate_content(prompt, generation_config=generation_config)
                    return response.text.strip()
                except Exception as e:
                    # More specific error message for model not found
                    if "404" in str(e) and "not found" in str(e):
                        available_models = ", ".join(self.list_models())
                        error_msg = f"Model '{model}' not found. Available models: {available_models}"
                        logging.error(error_msg)
                        return f"Error: {error_msg}"
                    else:
                        raise

        except Exception as e:
            error_msg = f"Gemini API call failed: {e}"
            logging.error(error_msg)

            # Return a more helpful message rather than None
            return f"Error: {error_msg}"

    def generate_stream(self, model_instance, prompt: str, generation_config: Dict[str, Any]) -> Iterator[str]:
        """Generate text using Gemini models with streaming enabled"""
        try:
            # Direct streaming without using chat
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            logging.error(f"Gemini streaming API call failed: {e}")
            yield f"Error: Gemini streaming API call failed: {e}"

    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for Gemini models"""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "top_k": 40
        }

    def validate_model(self, model_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate if a model name and parameters are compatible with Gemini"""
        issues = []
        try:
            # Remove 'models/' prefix if it exists
            if model_name.startswith("models/"):
                model_name = model_name.replace("models/", "")

            # Handle legacy model names
            if model_name == "gemini-pro":
                model_name = "gemini-1.5-pro"
                issues.append(f"Note: Model 'gemini-pro' will be mapped to '{model_name}'")
            elif model_name == "gemini-pro-vision":
                # Keep as is
                pass
            elif model_name == "gemini-2.0-pro":
                model_name = "gemini-1.5-pro"
                issues.append(f"Note: Model 'gemini-2.0-pro' will be mapped to '{model_name}'")
            elif model_name == "gemini-2.0-flash":
                # Check if actually available
                try:
                    available_models = self.list_models()
                    if "gemini-2.0-flash" not in available_models:
                        model_name = "gemini-1.5-flash"
                        issues.append(f"Note: Model 'gemini-2.0-flash' not found, will use '{model_name}'")
                except:
                    # If can't check, assume 1.5 is safer
                    model_name = "gemini-1.5-flash"
                    issues.append(f"Note: Unable to verify 'gemini-2.0-flash', will use '{model_name}'")

            # Check if model exists
            available_models = self.list_models()
            if model_name not in available_models:
                # Find the closest match
                if "gemini-1.5-pro" in available_models:
                    suggested_model = "gemini-1.5-pro"
                elif "gemini-1.5-flash" in available_models:
                    suggested_model = "gemini-1.5-flash"
                else:
                    suggested_model = available_models[0] if available_models else "gemini-1.5-pro"

                issues.append(f"Model '{model_name}' not found in Gemini's model list. Suggested alternative: '{suggested_model}'. Available models: {', '.join(available_models)}")
        except Exception as e:
            issues.append(f"Error validating model '{model_name}': {e}")

        # Validate parameters
        if "temperature" in params and not 0 <= params["temperature"] <= 1:
            issues.append("Temperature must be between 0 and 1")
        if "max_tokens" in params and params["max_tokens"] < 1:
            issues.append("max_tokens must be positive")
        if "top_p" in params and not 0 <= params["top_p"] <= 1:
            issues.append("top_p must be between 0 and 1")
        if "top_k" in params and params["top_k"] < 1:
            issues.append("top_k must be positive")

        return issues


def get_provider(provider_name: str) -> ModelProvider:
    """Factory function to get the appropriate provider instance"""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        # Add more providers as they are implemented
    }

    if provider_name.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}. Available providers: {', '.join(providers.keys())}")

    return providers[provider_name.lower()]()
