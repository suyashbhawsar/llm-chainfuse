import json
import yaml
import os
import sys
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Union
from model_providers import get_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LLMInference:
    def __init__(self,
                 provider: str = "openai",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 provider_options: Optional[Dict[str, Any]] = None,
                 **model_params):
        """
        Initialize LLM inference with pluggable provider support.

        Args:
            provider: The provider name (openai, anthropic, ollama, etc.)
            api_key: API key for the provider
            model: Default model to use
            provider_options: Additional provider-specific initialization options
            **model_params: Default model parameters (temperature, max_tokens, etc.)
        """
        self.provider_name = provider.lower()

        # Initialize the provider
        self.provider = get_provider(self.provider_name)

        # Initialize with provider-specific options
        provider_options = provider_options or {}
        self.provider.initialize(api_key=api_key, **provider_options)

        # Default model
        self.default_model = model

        # Initialize with empty default parameters
        self.default_params = {}

        # Override with user-provided parameters
        self.default_params.update({k: v for k, v in model_params.items() if v is not None})

        # Results storage for prompt chaining
        self.results = {}

        # Context data storage
        self.contexts = {}

    def add_contexts(self, context_files: Dict[str, str]) -> None:
        """
        Load and add context data as dynamic variables.

        Args:
            context_files: Dictionary mapping IDs to file paths
        """
        for context_id, file_path in context_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.contexts[context_id] = f.read().strip()
                logging.info(f"Loaded context for ID '{context_id}' from '{file_path}'.")
            except Exception as e:
                logging.error(f"Failed to load context file '{file_path}': {e}")
                raise

    def list_available_models(self) -> List[str]:
        """
        Fetch available models from the current provider.

        Returns:
            List of model names
        """
        return self.provider.list_models()

    def validate_models(self, file_path: str) -> None:
        """
        Validate models and parameters in the prompt file.

        Args:
            file_path: Path to prompt file
        """
        from model_providers import get_provider

        prompts = self.load_prompts(file_path)
        for p in prompts:
            # Get provider for this specific prompt
            provider_name = p.get("provider", self.provider_name)
            model_name = p.get("model", self.default_model)

            # Extract only the universal parameters plus any that are specifically set in the prompt
            # Don't include default params that weren't explicitly set in the prompt
            params = {}
            for k in ["temperature", "max_tokens", "top_p"]:
                if k in p:
                    params[k] = p.get(k)

            # Only add non-universal parameters if they're explicitly in the prompt
            for k in ["seed", "frequency_penalty", "presence_penalty", "logprobs"]:
                if k in p:
                    params[k] = p.get(k)

            try:
                # Initialize provider for this prompt
                prompt_provider = get_provider(provider_name)
                prompt_provider.initialize()

                # Validate through the appropriate provider
                issues = prompt_provider.validate_model(model_name, params)

                if issues:
                    for issue in issues:
                        logging.warning(issue)
                else:
                    logging.info(f"Model '{model_name}' (provider: {provider_name}) validation successful.")
            except Exception as e:
                logging.warning(f"Error validating model '{model_name}' with provider '{provider_name}': {e}")

        logging.info("Model validation completed.")

    def call_api(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Make an API call to the provider with optional parameter overrides.

        Args:
            prompt: The text prompt to send
            **kwargs: Override parameters for this specific call

        Returns:
            Generated text or None if an error occurred
        """
        # Determine which model to use
        model = kwargs.pop("model", self.default_model)

        # Start with empty parameters, only use explicit parameters
        params = {k: v for k, v in self.default_params.items() if v is not None}
        params.update({k: v for k, v in kwargs.items() if v is not None})

        # Call the provider's generate method
        return self.provider.generate(prompt, model, **params)

    def resolve_prompt(self, prompt: str) -> str:
        """
        Replace variables in a prompt with previously generated results or loaded context.

        Args:
            prompt: Raw prompt with placeholders

        Returns:
            Resolved prompt with placeholders replaced
        """
        if not prompt:
            return ""

        result = prompt
        for key, value in {**self.results, **self.contexts}.items():
            if value is None:
                value = ""  # Handle None values by replacing with empty string
            placeholder = f"{{{{ {key} }}}}"  # Example: {{ summary }} or {{ my_context }}
            result = result.replace(placeholder, str(value))
        return result

    def process_prompts(self, prompts: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Process prompts considering dependencies and custom parameters.

        Args:
            prompts: List of prompt configurations

        Returns:
            Dictionary of prompt IDs to results
        """
        from model_providers import get_provider

        results = {}

        # Separate independent and dependent prompts
        independent_prompts = [p for p in prompts if not any("{{" in p.get("prompt", "") for p in prompts)]
        dependent_prompts = [p for p in prompts if p not in independent_prompts]

        # Process independent prompts in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for p in independent_prompts:
                # Get the provider for this prompt
                provider_name = p.get("provider", self.provider_name)
                model_name = p.get("model", self.default_model)

                # Extract only parameters that are explicitly set in the prompt
                # Don't include default params that weren't explicitly set in the prompt
                prompt_params = {}
                for k, v in p.items():
                    if k not in ["prompt", "id", "provider", "model"]:
                        prompt_params[k] = v

                # Create a function that will process this prompt with the correct provider
                def process_with_provider(prompt_text, provider_name, model_name, params):
                    try:
                        prompt_provider = get_provider(provider_name)
                        prompt_provider.initialize()
                        return prompt_provider.generate(prompt_text, model_name, **params)
                    except Exception as e:
                        logging.error(f"Error with provider {provider_name}: {e}")
                        return None

                # Submit the task to the executor
                futures.append(executor.submit(
                    process_with_provider,
                    p["prompt"],
                    provider_name,
                    model_name,
                    prompt_params
                ))

            # Collect the results
            responses = [future.result() for future in futures]

        # Store the results
        for p, response in zip(independent_prompts, responses):
            if response is not None:
                self.results[p["id"]] = response
                results[p["id"]] = response
            else:
                logging.error(f"Failed to get response for prompt ID: {p['id']}")
                # Store empty response to prevent errors in dependent prompts
                self.results[p["id"]] = ""
                results[p["id"]] = ""

        # Process dependent prompts sequentially
        for p in dependent_prompts:
            # Get the provider for this prompt
            provider_name = p.get("provider", self.provider_name)
            model_name = p.get("model", self.default_model)

            # Extract only parameters that are explicitly set in the prompt
            prompt_params = {}
            for k, v in p.items():
                if k not in ["prompt", "id", "provider", "model"]:
                    prompt_params[k] = v

            # Resolve the prompt text
            resolved_prompt = self.resolve_prompt(p["prompt"])

            try:
                # Initialize the correct provider
                prompt_provider = get_provider(provider_name)
                prompt_provider.initialize()

                # Generate the response
                response = prompt_provider.generate(resolved_prompt, model_name, **prompt_params)

                if response is not None:
                    self.results[p["id"]] = response
                    results[p["id"]] = response
                else:
                    logging.error(f"Failed to get response for prompt ID: {p['id']}")
                    # Store empty response to prevent further errors
                    self.results[p["id"]] = ""
                    results[p["id"]] = ""
            except Exception as e:
                logging.error(f"Error processing prompt {p['id']} with provider {provider_name}: {e}")
                self.results[p["id"]] = ""
                results[p["id"]] = ""

        return results

    def load_prompts(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load prompts from YAML or JSON file.

        Args:
            file_path: Path to prompt file

        Returns:
            List of prompt configurations
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(file).get("prompts", [])
            elif file_path.endswith('.json'):
                return json.load(file).get("prompts", [])
            else:
                raise ValueError("Unsupported file format. Use YAML or JSON.")

    def run(self, file_path: str, output_file: Optional[str] = None, cli_args=None) -> Dict[str, str]:
        """
        Run inference on prompts from the given file and save results.

        Args:
            file_path: Path to prompt file
            output_file: Optional path to save results
            cli_args: Arguments passed from CLI to control behavior

        Returns:
            Dictionary of prompt IDs to results
        """
        prompts = self.load_prompts(file_path)
        if not prompts:
            raise ValueError("No prompts found in the input file.")

        # Store original prompt order to maintain sequence in output
        prompt_ids_ordered = [p.get("id") for p in prompts]

        # Check if the file has global print settings
        print_config = self._extract_print_config(file_path)

        # Get unique providers needed for this run
        providers_needed = set([p.get("provider", self.provider_name).lower() for p in prompts])
        logging.info(f"Providers needed for this run: {', '.join(providers_needed)}")

        # Only initialize default provider if actually used
        if self.provider_name not in providers_needed:
            logging.info(f"Default provider '{self.provider_name}' not used in this run")

        # Process the prompts
        results = self.process_prompts(prompts)

        # Determine if CLI print mode is active
        cli_print_mode = cli_args and hasattr(cli_args, 'print') and cli_args.print is not None

        # If print_config exists in YAML and we're not using CLI print mode, show results
        if print_config and not cli_print_mode:
            print("\n=== LLM INFERENCE RESULTS (from YAML/JSON) ===\n")

            if print_config.get("print_all", False):
                # Print all results in the original order
                for prompt_id in prompt_ids_ordered:
                    if prompt_id in results:
                        print(f"\n== RESULT: {prompt_id} ==\n")
                        print(results[prompt_id])
                        print("\n" + "="*50 + "\n")
            elif print_config.get("print_ids"):
                # Print only the specified prompt IDs
                for prompt_id in print_config["print_ids"]:
                    if prompt_id in results:
                        print(f"\n== RESULT: {prompt_id} ==\n")
                        print(results[prompt_id])
                        print("\n" + "="*50 + "\n")
                    else:
                        print(f"Warning: No result found for prompt ID '{prompt_id}'")
        elif not cli_print_mode:
            # In non-print mode, just show completion status for each prompt
            print("\n=== LLM INFERENCE STATUS ===\n")
            for prompt_id in prompt_ids_ordered:
                if prompt_id in results and results[prompt_id]:
                    print(f"✅ Prompt '{prompt_id}': Success")
                else:
                    print(f"❌ Prompt '{prompt_id}': Failed or empty response")
            print("\nUse --debug flag for more details or --print to see results\n")

        # Save results to file if requested
        if output_file:
            output_path = output_file
        else:
            output_path = file_path.replace('.yaml', '_output.json').replace('.json', '_output.json')
            if output_path == file_path:  # Safeguard against overwriting input file
                output_path = f"{file_path}_output.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        logging.info(f"Results saved to {output_path}")
        return results

    def _extract_print_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract printing configuration from the prompt file.

        Args:
            file_path: Path to prompt file

        Returns:
            Dictionary with printing configuration or None
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(file)
            elif file_path.endswith('.json'):
                data = json.load(file)
            else:
                return None

        # Check if the file has a global print configuration
        if "print" in data:
            return data["print"]
        return None
