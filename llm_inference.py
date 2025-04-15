import json
import yaml
import os
import sys
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from model_providers import get_provider
import re
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


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

        # Store resolved prompts for display
        self.resolved_prompts = {}

        # Context data storage
        self.contexts = {}

    def get_nested_value(self, d: Dict, path: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-separated path.

        Args:
            d: Dictionary to search in
            path: Dot-separated path to the value (e.g., 'a.b.c')

        Returns:
            The value at the path, or empty string if not found
        """
        keys = [k.strip() for k in path.split('.')]  # Strip whitespace from each key
        current = d
        logging.debug(f"Getting nested value for path: {path}")

        for key in keys:
            if isinstance(current, dict):
                if key in current:
                    current = current.get(key)
                else:
                    logging.debug(f"Key '{key}' not found in dictionary")
                    return ''
            else:
                logging.debug(f"Current value is not a dictionary")
                return ''

        return current

    def add_contexts(self, context_files: Dict[str, str]) -> None:
        """
        Load and add context data as dynamic variables.

        Args:
            context_files: Dictionary mapping IDs to file paths
        """
        for context_id, file_path in context_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if file_path.endswith(('.yaml', '.yml')):
                        try:
                            yaml_data = yaml.safe_load(content)
                            # Store the entire YAML data structure
                            self.contexts[context_id] = yaml_data
                            logging.info(f"Loaded YAML context for ID '{context_id}' from '{file_path}'")
                        except yaml.YAMLError as e:
                            logging.error(f"Failed to parse YAML file '{file_path}': {e}")
                            raise
                    else:
                        # For non-YAML files, store as plain text
                        self.contexts[context_id] = content.strip()
                        logging.info(f"Loaded text context for ID '{context_id}' from '{file_path}'")
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

    def call_api(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        Make an API call to the provider with optional parameter overrides.

        Args:
            prompt: The text prompt to send
            stream: Whether to stream the response
            **kwargs: Override parameters for this specific call

        Returns:
            If stream=False: Generated text or None if an error occurred
            If stream=True: Iterator yielding response chunks
        """
        # Determine which model to use
        model = kwargs.pop("model", self.default_model)

        # Start with empty parameters, only use explicit parameters
        params = {k: v for k, v in self.default_params.items() if v is not None}
        params.update({k: v for k, v in kwargs.items() if v is not None})

        # Call the provider's generate method
        return self.provider.generate(prompt, model, stream=stream, **params)

    def _extract_dependencies(self, prompt_text: str, prompt_map: Dict[str, Dict]) -> List[str]:
        """
        Extract prompt IDs mentioned as dependencies in a prompt string.

        Args:
            prompt_text: The prompt text to parse.
            prompt_map: A dictionary mapping prompt IDs to their configurations.

        Returns:
            A list of prompt IDs that are dependencies.
        """
        if not prompt_text:
            return []
        pattern = r"\{\{\s*([^}]+)\s*\}\}"
        dependencies = re.findall(pattern, prompt_text)
        # Filter to only include valid prompt IDs and ignore context variables
        prompt_dependencies = [dep.strip() for dep in dependencies if dep.strip() in prompt_map]
        return prompt_dependencies

    def _find_all_dependencies(self, target_id: str, prompts_dict: Dict[str, Dict]) -> set:
        """
        Recursively find all prompt IDs required to run the target ID.

        Args:
            target_id: The final prompt ID to run.
            prompts_dict: Dictionary mapping prompt IDs to their configurations.

        Returns:
            A set of all required prompt IDs (including the target_id).
        """
        required_ids = {target_id}
        if target_id not in prompts_dict:
            raise ValueError(f"Target prompt ID '{target_id}' not found in the input file.")

        prompt_config = prompts_dict[target_id]
        prompt_text = prompt_config.get("prompt", "")
        direct_dependencies = self._extract_dependencies(prompt_text, prompts_dict)

        for dep_id in direct_dependencies:
            if dep_id not in required_ids:  # Avoid infinite loops in case of circular dependencies (though process_prompts handles this)
                required_ids.update(self._find_all_dependencies(dep_id, prompts_dict))

        return required_ids

    def resolve_prompt(self, prompt: str, prompt_id: Optional[str] = None) -> str:
        """
        Replace variables in a prompt with previously generated results or loaded context.
        Supports nested YAML paths using dot notation (e.g., {{ profile.introduction.fullName }}).

        Args:
            prompt: Raw prompt with placeholders
            prompt_id: Optional ID to store the resolved prompt for display

        Returns:
            Resolved prompt with placeholders replaced
        """
        if not prompt:
            return ""

        result = prompt
        logging.debug(f"Resolving prompt with ID: {prompt_id}")

        # First pass: Handle nested YAML paths
        for key, value in self.contexts.items():
            if isinstance(value, dict):
                # Match the entire placeholder pattern including the context key
                pattern = r"\{\{\s*" + re.escape(key) + r"\.([^}]+)\s*\}\}"
                matches = re.finditer(pattern, result)
                for match in matches:
                    full_match = match.group(0)
                    nested_path = match.group(1)
                    nested_value = self.get_nested_value(value, nested_path)
                    if nested_value != '':
                        result = result.replace(full_match, str(nested_value))
                    else:
                        logging.warning(f"Could not resolve nested path '{nested_path}' in context '{key}'")

        # Second pass: Handle direct key replacements for backward compatibility
        for key, value in {**self.results, **self.contexts}.items():
            if value is None:
                value = ""  # Handle None values by replacing with empty string
            if isinstance(value, (str, int, float, bool)):  # Only replace if value is a simple type
                spaced_placeholder = f"{{{{ {key} }}}}"
                unspaced_placeholder = f"{{{{{key}}}}}"
                if spaced_placeholder in result or unspaced_placeholder in result:
                    result = result.replace(spaced_placeholder, str(value))
                    result = result.replace(unspaced_placeholder, str(value))

        # Store the resolved prompt if ID is provided
        if prompt_id and "{{" in prompt:
            self.resolved_prompts[prompt_id] = result

        return result

    def process_prompts(self, prompts: List[Dict[str, Any]],
                        stream_callback: Optional[Callable[[str, str], None]] = None) -> Dict[str, str]:
        """
        Process prompts considering dependencies and custom parameters.

        Args:
            prompts: List of prompt configurations
            stream_callback: Optional callback function that takes prompt_id and result
                           to handle streaming output

        Returns:
            Dictionary of prompt IDs to results
        """
        from model_providers import get_provider
        import re
        import time

        results = {}
        timing_info = {}  # Store timing information for each prompt

        # Build prompt map for easy lookup
        prompt_map = {p["id"]: p for p in prompts}

        # Extract dependency pattern - matches {{ variable_name }}
        def extract_dependencies(prompt_text):
            return self._extract_dependencies(prompt_text, prompt_map)

        # Build dependency graph and identify independent prompts
        dependency_graph = {}
        independent_prompts = []

        for p in prompts:
            prompt_id = p["id"]
            prompt_text = p.get("prompt", "")
            dependencies = extract_dependencies(prompt_text)

            # Filter to only include prompt IDs that exist in our prompt list
            # We use the filtered 'prompts' list here, not the global prompt_map
            prompt_dependencies = [dep for dep in dependencies if dep in prompt_map]

            if not prompt_dependencies:
                independent_prompts.append(p)
            else:
                dependency_graph[prompt_id] = prompt_dependencies

        logging.info(f"Independent prompts to run: {[p['id'] for p in independent_prompts]}")
        logging.info(f"Dependency graph for this run: {dependency_graph}")

        # Process independent prompts in parallel
        with ThreadPoolExecutor() as executor:
            future_to_prompt = {}

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

                # Resolve the prompt text first
                resolved_prompt = self.resolve_prompt(p["prompt"], p["id"])
                logging.debug(f"Resolved prompt for {p['id']}: {resolved_prompt}")

                # Create a function that will process this prompt with the correct provider
                def process_with_provider(prompt_config, resolved_text, provider_name, model_name, params):
                    try:
                        start_time = time.time()
                        # Re-get provider instance within the thread
                        prompt_provider_instance = get_provider(provider_name)
                        prompt_provider_instance.initialize() # Ensure initialized in thread
                        response = prompt_provider_instance.generate(resolved_text, model_name, **params)
                        end_time = time.time()
                        return prompt_config["id"], response, end_time - start_time
                    except Exception as e:
                        logging.error(f"Error processing prompt '{prompt_config['id']}' with provider {provider_name}: {e}")
                        return prompt_config["id"], None, 0

                # Submit the task to the executor
                future = executor.submit(
                    process_with_provider,
                    p,
                    resolved_prompt,
                    provider_name,
                    model_name,
                    prompt_params
                )
                future_to_prompt[future] = p

            # Process results as they come in
            for future in as_completed(future_to_prompt):
                prompt_id, response, duration = future.result()
                prompt = future_to_prompt[future]

                if response is not None:
                    self.results[prompt_id] = response
                    results[prompt_id] = response
                    timing_info[prompt_id] = duration

                    # Call streaming callback if provided
                    if stream_callback:
                        stream_callback(prompt_id, response)
                else:
                    logging.error(f"Failed to get response for prompt ID: {prompt_id}")
                    # Store empty response to prevent errors in dependent prompts
                    self.results[prompt_id] = ""
                    results[prompt_id] = ""
                    timing_info[prompt_id] = 0

                    # Call streaming callback with empty result if provided
                    if stream_callback:
                        stream_callback(prompt_id, "")

        # Process dependent prompts as soon as their dependencies are resolved
        remaining_dependent_prompts = list(dependency_graph.keys())

        while remaining_dependent_prompts:
            # Find prompts whose dependencies are all resolved
            ready_prompts = []
            for prompt_id in remaining_dependent_prompts:
                dependencies = dependency_graph[prompt_id]
                if all(dep in self.results for dep in dependencies):
                    ready_prompts.append(prompt_id)

            if not ready_prompts:
                # If no prompts are ready but we still have remaining prompts,
                # there might be a circular dependency or missing prompt
                logging.error(f"Unable to resolve dependencies for: {remaining_dependent_prompts}")
                for prompt_id in remaining_dependent_prompts:
                    self.results[prompt_id] = f"Error: Could not resolve dependencies {dependency_graph[prompt_id]}"
                    results[prompt_id] = self.results[prompt_id]
                    timing_info[prompt_id] = 0
                    if stream_callback:
                        stream_callback(prompt_id, self.results[prompt_id])
                break

            # Process all ready prompts in parallel
            with ThreadPoolExecutor() as executor:
                future_to_prompt = {}

                for prompt_id in ready_prompts:
                    p = prompt_map[prompt_id]
                    # Get the provider for this prompt
                    provider_name = p.get("provider", self.provider_name)
                    model_name = p.get("model", self.default_model)

                    # Extract parameters
                    prompt_params = {}
                    for k, v in p.items():
                        if k not in ["prompt", "id", "provider", "model"]:
                            prompt_params[k] = v

                    # Resolve the prompt text
                    resolved_prompt = self.resolve_prompt(p["prompt"], p["id"])
                    logging.debug(f"Resolved prompt for {p['id']}: {resolved_prompt}")

                    # Submit the task to the executor
                    def process_resolved_prompt(prompt_id, resolved_text, provider_name, model_name, params):
                        try:
                            start_time = time.time()
                            # Re-get provider instance within the thread
                            prompt_provider_instance = get_provider(provider_name)
                            prompt_provider_instance.initialize() # Ensure initialized in thread
                            response = prompt_provider_instance.generate(resolved_text, model_name, **params)
                            end_time = time.time()
                            return prompt_id, response, end_time - start_time
                        except Exception as e:
                            logging.error(f"Error processing prompt '{prompt_id}' with provider {provider_name}: {e}")
                            return prompt_id, None, 0

                    future = executor.submit(
                        process_resolved_prompt,
                        prompt_id,
                        resolved_prompt,
                        provider_name,
                        model_name,
                        prompt_params
                    )
                    future_to_prompt[future] = p

                # Process results as they come in
                for future in as_completed(future_to_prompt):
                    prompt_id, response, duration = future.result()
                    prompt = future_to_prompt[future]

                    if response is not None:
                        self.results[prompt_id] = response
                        results[prompt_id] = response
                        timing_info[prompt_id] = duration

                        # Call streaming callback if provided
                        if stream_callback:
                            stream_callback(prompt_id, response)
                    else:
                        logging.error(f"Failed to get response for prompt ID: {prompt_id}")
                        self.results[prompt_id] = ""
                        results[prompt_id] = ""
                        timing_info[prompt_id] = 0

                        # Call streaming callback with empty result
                        if stream_callback:
                            stream_callback(prompt_id, "")

            # Remove processed prompts from the remaining list
            for prompt_id in ready_prompts:
                remaining_dependent_prompts.remove(prompt_id)

        # Log timing information if in debug mode
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            logging.info("\n=== Prompt Processing Times ===")
            for prompt_id, duration in timing_info.items():
                logging.info(f"{prompt_id}: {duration:.2f} seconds")

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
                data = yaml.safe_load(file)
                if not isinstance(data, dict) or "prompts" not in data:
                    raise ValueError("YAML file must contain a top-level 'prompts' key with a list of prompts.")
                return data.get("prompts", [])
            elif file_path.endswith('.json'):
                data = json.load(file)
                if not isinstance(data, dict) or "prompts" not in data:
                    raise ValueError("JSON file must contain a top-level 'prompts' key with a list of prompts.")
                return data.get("prompts", [])
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
        all_prompts_list = self.load_prompts(file_path)
        if not all_prompts_list:
            raise ValueError("No prompts found in the input file.")

        # Create a dictionary for quick lookup
        all_prompts_dict = {p.get("id"): p for p in all_prompts_list if p.get("id")}

        # Determine which prompts to run
        prompts_to_run_list = all_prompts_list
        target_run_id = cli_args.run_id if cli_args and hasattr(cli_args, 'run_id') else None

        if target_run_id:
            if target_run_id not in all_prompts_dict:
                raise ValueError(f"Prompt ID '{target_run_id}' specified with --run-id not found in '{file_path}'.")

            # Find all dependencies for the target ID
            required_ids = self._find_all_dependencies(target_run_id, all_prompts_dict)
            logging.info(f"Running target ID '{target_run_id}' and its dependencies: {required_ids}")

            # Filter the list of prompts to include only the required ones
            prompts_to_run_list = [p for p in all_prompts_list if p.get("id") in required_ids]

            # Check if all required prompts were found (sanity check)
            found_ids = {p.get("id") for p in prompts_to_run_list}
            if found_ids != required_ids:
                 missing = required_ids - found_ids
                 logging.warning(f"Could not find configurations for required prompt IDs: {missing}")
                 # Continue with the found prompts, process_prompts will handle errors for missing ones
        else:
             logging.info(f"Running all prompts from '{file_path}'")


        # Store original prompt order (from the filtered list) to maintain sequence in output
        prompt_ids_ordered = [p.get("id") for p in prompts_to_run_list]

        # Check if the file has global print settings
        print_config = self._extract_print_config(file_path)

        # Get unique providers needed for this specific run
        providers_needed = set([p.get("provider", self.provider_name).lower() for p in prompts_to_run_list])
        logging.info(f"Providers needed for this run: {', '.join(providers_needed)}")

        # Only initialize default provider if actually used
        if self.provider_name not in providers_needed:
            logging.info(f"Default provider '{self.provider_name}' not used in this run")

        # Determine CLI modes
        cli_print_mode = cli_args and hasattr(cli_args, 'print') and cli_args.print is not None
        cli_silent_mode = cli_args and hasattr(cli_args, 'silent') and cli_args.silent
        cli_output_file = cli_args and hasattr(cli_args, 'output') and cli_args.output
        cli_batch_mode = cli_args and hasattr(cli_args, 'batch') and cli_args.batch
        cli_show_prompts = cli_args and hasattr(cli_args, 'show_prompts') and cli_args.show_prompts
        cli_debug_mode = cli_args and hasattr(cli_args, 'debug') and cli_args.debug
        cli_verbose_mode = cli_args and hasattr(cli_args, 'verbose') and cli_args.verbose

        # Create prompt mapping for display (only for the prompts being run)
        prompts_dict_for_run = {p.get("id"): p for p in prompts_to_run_list}

        # Define output callback function for immediate display (default behavior)
        output_callback = None
        if not cli_batch_mode and not cli_silent_mode:
            def immediate_output(prompt_id, result):
                # Only print if the prompt ID was actually part of the run
                if prompt_id not in prompts_dict_for_run:
                    return

                if not result: # Don't print empty results unless in debug mode maybe?
                    logging.debug(f"Skipping print for empty result of {prompt_id}")
                    # Print errors even if empty
                    if "Error:" in str(result):
                         print(f"\n== RESULT: {prompt_id} ==\n")
                         print(result)
                         print("\n" + "="*50 + "\n")
                    return

                print(f"\n== RESULT: {prompt_id} ==\n")

                # Print prompt if requested
                if cli_show_prompts and prompt_id in prompts_dict_for_run:
                    prompt_config = prompts_dict_for_run[prompt_id]
                    prompt_text = prompt_config.get("prompt", "")

                    # If this is a dependent prompt (with variables), show the resolved version
                    if "{{" in prompt_text and prompt_id in self.resolved_prompts:
                        print("Original Prompt:")
                        print(prompt_text)
                        print("\nResolved Prompt:")
                        print(self.resolved_prompts[prompt_id])
                    else:
                        print("Prompt:")
                        print(prompt_text)
                    print("\nResponse:")

                # Print debug info if enabled
                if cli_debug_mode and prompt_id in prompts_dict_for_run:
                    prompt_config = prompts_dict_for_run[prompt_id]
                    if cli_verbose_mode:
                        print("Configuration:")
                        for key, value in prompt_config.items():
                            if key != "prompt" and key != "id":
                                print(f"  {key}: {value}")
                        if not cli_show_prompts:  # Don't duplicate prompt if already shown
                            print("\nPrompt:")
                            print(prompt_config.get("prompt", ""))
                        print("\nResult:")
                    else:
                        print("Configuration:", end=" ")
                        config_str = ", ".join(f"{k}={v}" for k, v in prompt_config.items()
                                             if k != "prompt" and k != "id")
                        print(config_str)
                        print("\nResult:")

                print(result)
                print("\n" + "="*50 + "\n")

            output_callback = immediate_output

            # Print header for immediate output mode
            if not cli_silent_mode:
                print("\n=== LLM INFERENCE RESULTS ===\n")

        # Process the selected prompts
        results = self.process_prompts(prompts_to_run_list, output_callback)

        # If in batch mode and not in silent mode, print results
        if cli_batch_mode and not cli_silent_mode:
            # Determine which IDs to print based on CLI or YAML config
            ids_to_print_final = []
            if cli_print_mode:
                # CLI overrides YAML print config
                ids_to_print_final = cli_args.print
                print("\n=== LLM INFERENCE RESULTS (CLI specified) ===\n")
            elif print_config:
                 # Use YAML print config if no CLI print args
                print("\n=== LLM INFERENCE RESULTS (from YAML/JSON) ===\n")
                if print_config.get("print_all", False):
                    # Print all results *that were actually run*
                    ids_to_print_final = list(results.keys())
                elif print_config.get("print_ids"):
                    # Print only the specified prompt IDs *if they were run*
                    ids_to_print_final = [pid for pid in print_config["print_ids"] if pid in results]
            else:
                # Default: print all results *that were run*
                ids_to_print_final = list(results.keys())
                print("\n=== LLM INFERENCE RESULTS ===\n")

            # Filter ids_to_print_final further if --run-id was used
            if target_run_id:
                # If --run-id was used, only print that specific ID's result
                # unless other IDs were explicitly requested via --print
                if not cli_print_mode: # If user didn't specify --print, just show the target
                    ids_to_print_final = [target_run_id] if target_run_id in results else []
                else: # If user specified --print, respect that, but only show run IDs
                    ids_to_print_final = [pid for pid in cli_args.print if pid in results]

            # Sort the IDs to match the original order in the *filtered* prompt list
            if prompt_ids_ordered:
                sorted_ids = [pid for pid in prompt_ids_ordered if pid in ids_to_print_final]
                # Add any remaining requested IDs that might not be in the original order (e.g., if --print specifies different order)
                sorted_ids.extend([pid for pid in ids_to_print_final if pid not in sorted_ids])
                ids_to_print = sorted_ids
            else:
                 ids_to_print = ids_to_print_final # Fallback if ordering fails


            # Display results in the determined order
            for prompt_id in ids_to_print:
                if prompt_id in results:
                    print(f"\n== RESULT: {prompt_id} ==\n")

                    # Print prompt if requested
                    if cli_show_prompts and prompt_id in prompts_dict_for_run:
                        prompt_config = prompts_dict_for_run[prompt_id]
                        prompt_text = prompt_config.get("prompt", "")

                        # If this is a dependent prompt (with variables), show the resolved version
                        if "{{" in prompt_text and prompt_id in self.resolved_prompts:
                            print("Original Prompt:")
                            print(prompt_text)
                            print("\nResolved Prompt:")
                            print(self.resolved_prompts[prompt_id])
                        else:
                            print("Prompt:")
                            print(prompt_text)
                        print("\nResponse:")

                    # Print debug info if enabled
                    if cli_debug_mode and prompt_id in prompts_dict_for_run:
                        prompt_config = prompts_dict_for_run[prompt_id]
                        if cli_verbose_mode:
                            print("Configuration:")
                            for key, value in prompt_config.items():
                                if key != "prompt" and key != "id":
                                    print(f"  {key}: {value}")
                            if not cli_show_prompts:  # Don't duplicate prompt if already shown
                                print("\nPrompt:")
                                print(prompt_config.get("prompt", ""))
                            print("\nResult:")
                        else:
                            print("Configuration:", end=" ")
                            config_str = ", ".join(f"{k}={v}" for k, v in prompt_config.items()
                                                if k != "prompt" and k != "id")
                            print(config_str)
                            print("\nResult:")

                    print(results[prompt_id])
                    print("\n" + "="*50 + "\n")
                else:
                    print(f"Warning: No result found for prompt ID '{prompt_id}' (it might not have been run or failed).")

        # Save results to file if requested
        if output_file:
            # Save only the results that were actually generated in this run
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            logging.info(f"Results for prompts {list(results.keys())} saved to {output_file}")
        elif cli_output_file and not output_file: # Check if output was requested via CLI but not passed directly (shouldn't happen with current main.py but good practice)
             with open(cli_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
             logging.info(f"Results for prompts {list(results.keys())} saved to {cli_output_file}")


        return results

    def _extract_print_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract printing configuration from the prompt file.

        Args:
            file_path: Path to prompt file

        Returns:
            Dictionary with printing configuration or None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(file)
                elif file_path.endswith('.json'):
                    data = json.load(file)
                else:
                    return None

            # Check if the file has a global print configuration
            if isinstance(data, dict) and "print" in data:
                return data["print"]
        except Exception as e:
            logging.warning(f"Could not read or parse print config from '{file_path}': {e}")
            return None
        return None
