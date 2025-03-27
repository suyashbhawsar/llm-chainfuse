#!/usr/bin/env python3
import argparse
import json
import logging
import os
import io
import sys
from llm_inference import LLMInference
from model_providers import get_provider

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Multi-provider LLM Inference Tool")

    # Provider configuration
    provider_group = parser.add_argument_group("Provider Configuration")
    provider_group.add_argument("-p", "--provider", type=str, default="openai",
                             help="LLM provider (openai, anthropic, ollama)")
    provider_group.add_argument("-k", "--api-key", type=str,
                             help="API key for the selected provider (defaults to environment variable)")
    provider_group.add_argument("--provider-options", type=str,
                             help="JSON string of additional provider options")

    # Main arguments
    parser.add_argument("input", nargs="?", type=str,
                      help="Path to input file (YAML/JSON) or direct text prompt")
    parser.add_argument("prompt_args", nargs="*",
                      help="Additional text for prompt (concatenated with input)")
    parser.add_argument("-q", "--prompt", type=str,
                      help="Explicit direct text prompt (bypasses file input)")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o",
                      help="Model to use for inference")
    parser.add_argument("-t", "--temperature", type=float,
                      help="Temperature for randomness (0.0-2.0)")
    parser.add_argument("-n", "--max_tokens", type=int,
                      help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float,
                      help="Top-p sampling (0.0-1.0)")
    # Optional OpenAI-specific parameters
    openai_group = parser.add_argument_group("OpenAI-specific Parameters")
    openai_group.add_argument("-s", "--seed", type=int,
                           help="Seed for reproducibility (OpenAI only)")
    openai_group.add_argument("-f", "--frequency_penalty", type=float,
                           help="Frequency penalty (OpenAI only)")
    openai_group.add_argument("--presence_penalty", type=float,
                           help="Presence penalty (OpenAI only)")
    openai_group.add_argument("--logprobs", type=int,
                          help="Log probabilities (OpenAI only)")

    # Output control
    parser.add_argument("-o", "--output", type=str,
                      help="Path to output file (if not provided, results will not be saved to a file)")
    parser.add_argument("--silent", action="store_true",
                      help="Hide output in terminal")
    parser.add_argument("--print", nargs="*",
                      help="Print specific results (specify prompt IDs). If not specified, all results are printed by default.")
    parser.add_argument("--show-prompts", action="store_true",
                      help="Show the actual prompts sent to the LLM (with resolved variables)")

    # Context files
    parser.add_argument("-c", "--context", nargs="*",
                      help="List of context files (ID:path). Example: id1:context1.txt id2:context2.txt")

    # Utility functions
    utility_group = parser.add_argument_group("Utility Functions")
    utility_group.add_argument("-l", "--list-models", action="store_true",
                            help="List available models for the selected provider")
    utility_group.add_argument("--list-providers", action="store_true",
                            help="List available providers")
    utility_group.add_argument("--validate-models", action="store_true",
                            help="Validate models & parameters in the prompt file")
    utility_group.add_argument("-d", "--debug", action="store_true",
                            help="Enable debug mode with detailed output about LLM inference")
    utility_group.add_argument("-v", "--verbose", action="store_true",
                            help="Enable more verbose debug output (use with --debug)")

    args = parser.parse_args()

    # Set up logging based on debug flags
    if args.debug:
        log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s" if args.verbose else "%(asctime)s - %(levelname)s - %(message)s"
        # Reset the root logger and configure it with DEBUG level
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        # In non-debug mode, silence logging output
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.ERROR, format="%(message)s")

    # Handle list providers
    if args.list_providers:
        try:
            from model_providers import get_provider
            # Get a list of provider names by attempting to initialize each
            provider_names = []
            for provider_name in ["openai", "anthropic", "ollama"]:
                try:
                    get_provider(provider_name)
                    provider_names.append(provider_name)
                except (ImportError, ModuleNotFoundError):
                    pass
                except Exception:
                    pass

            print("\nAvailable Providers:")
            for provider in provider_names:
                print(f"- {provider}")
            return
        except Exception as e:
            print(f"Error listing providers: {e}")
            return

    # Parse provider options if provided
    provider_options = {}
    if args.provider_options:
        try:
            provider_options = json.loads(args.provider_options)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in provider options: {args.provider_options}")
            return

    # Initialize LLMInference with the selected provider
    try:
        # Determine the appropriate environment variable based on provider
        if not args.api_key:
            if args.provider.lower() == "openai":
                env_var = "OPENAI_API_KEY"
            elif args.provider.lower() == "anthropic":
                env_var = "ANTHROPIC_API_KEY"
            else:
                env_var = None

            if env_var and os.environ.get(env_var) and args.debug:
                print(f"Using API key from environment variable: {env_var}")

        llm = LLMInference(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            provider_options=provider_options,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty
        )
    except ValueError as e:
        print(f"Error initializing provider: {e}")
        return
    except ImportError as e:
        if args.provider.lower() == "anthropic":
            print("Error: Anthropic package not installed.")
            print("Install it with: pip install anthropic")
        elif args.provider.lower() == "openai":
            print("Error: OpenAI package not installed or incompatible version.")
            print("Install it with: pip install openai>=1.0.0")
        else:
            print(f"Error: Required package for provider '{args.provider}' is not installed.")
            print(f"Details: {e}")
        return

    # Load context files if provided
    if args.context:
        context_files = {}
        for context_pair in args.context:
            try:
                context_id, file_path = context_pair.split(":", 1)
                context_files[context_id] = file_path
            except ValueError:
                print(f"Invalid context argument format: '{context_pair}'. Expected format: ID:path.")
                return
        llm.add_contexts(context_files)

    # Handle list models functionality
    if args.list_models:
        models = llm.list_available_models()
        print(f"\nAvailable Models for {args.provider.capitalize()}:")
        for model in models:
            print(f"- {model}")
        return

    # Handle validate models functionality
    if args.validate_models:
        if not args.input:
            print("Error: Please provide a YAML/JSON file for model validation.")
            return
        llm.validate_models(args.input)
        return

    # Check if we're in direct prompt mode
    in_direct_prompt_mode = False
    direct_prompt_text = ""

    # Explicit --prompt argument takes precedence
    if args.prompt:
        in_direct_prompt_mode = True
        direct_prompt_text = args.prompt
    # Otherwise, check if input is a file or direct prompt
    elif args.input:
        # If additional prompt arguments were provided, treat input as part of the prompt
        if args.prompt_args:
            in_direct_prompt_mode = True
            direct_prompt_text = args.input + " " + " ".join(args.prompt_args)
        else:
            # Check if input is a valid file
            if os.path.isfile(args.input):
                # Input is a file, will process as YAML/JSON
                pass
            else:
                # Input is not a file, treat as direct prompt
                in_direct_prompt_mode = True
                direct_prompt_text = args.input

    # Handle YAML/JSON prompt execution
    if not in_direct_prompt_mode and args.input:
        try:
            # Load the prompts first to get the correct order for display
            loaded_prompts = []
            try:
                loaded_prompts = llm.load_prompts(args.input)
                prompt_ids_ordered = [p.get("id") for p in loaded_prompts]
            except Exception as e:
                logging.warning(f"Could not load prompts for ordering: {e}")

            # Pass args to run method to control behavior
            results = llm.run(args.input, args.output, cli_args=args)

            # Skip printing if --silent is provided
            if args.silent:
                if args.output:
                    print(f"Results saved to {args.output}")
                return

            # Determine which IDs to print
            if hasattr(args, 'print') and args.print:
                ids_to_print = args.print
            else:
                ids_to_print = list(results.keys())

            print("\n=== LLM INFERENCE RESULTS ===\n")

            # Create a mapping of prompts for debug info
            original_prompts = {}
            if args.debug or args.show_prompts:
                original_prompts = {p.get("id"): p for p in loaded_prompts}

            # Sort the IDs to match the original order in the YAML/JSON
            if loaded_prompts:
                # Filter ids_to_print to include only those in prompt_ids_ordered
                # and maintain the order from the original file
                sorted_ids = [pid for pid in prompt_ids_ordered if pid in ids_to_print]
                # Add any remaining IDs that might not be in prompt_ids_ordered
                sorted_ids.extend([pid for pid in ids_to_print if pid not in prompt_ids_ordered])
                ids_to_print = sorted_ids

            # Display results in the correct order
            for prompt_id in ids_to_print:
                if prompt_id in results:
                    print(f"\n== RESULT: {prompt_id} ==\n")

                    # Print prompt if requested
                    if args.show_prompts and prompt_id in original_prompts:
                        prompt_config = original_prompts[prompt_id]
                        prompt_text = prompt_config.get("prompt", "")

                        # If this is a dependent prompt (with variables), show the resolved version
                        if "{{" in prompt_text and prompt_id in llm.resolved_prompts:
                            print("Original Prompt:")
                            print(prompt_text)
                            print("\nResolved Prompt:")
                            print(llm.resolved_prompts[prompt_id])
                        else:
                            print("Prompt:")
                            print(prompt_text)
                        print("\nResponse:")

                    # Print debug info if enabled
                    if args.debug and prompt_id in original_prompts:
                        prompt_config = original_prompts[prompt_id]
                        if args.verbose:
                            print("Configuration:")
                            for key, value in prompt_config.items():
                                if key != "prompt" and key != "id":
                                    print(f"  {key}: {value}")
                            if not args.show_prompts:  # Don't duplicate prompt if already shown
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
                    print(f"Warning: No result found for prompt ID '{prompt_id}'")
        except Exception as e:
            print(f"Error running inference: {e}")
    elif in_direct_prompt_mode:
        # Direct prompt mode
        try:
            # Skip printing if --silent is provided
            if not args.silent:
                # Print the prompt if debug or show_prompts is enabled
                if args.debug or args.show_prompts:
                    print("\n=== DIRECT PROMPT ===\n")
                    print(direct_prompt_text)
                    print("\n=== RESPONSE ===\n")

            # Call the LLM directly with the provided prompt
            result = llm.call_api(direct_prompt_text)

            # Print the result if not in silent mode
            if not args.silent:
                if result:
                    print(result)
                else:
                    print("Error: No response received from the model.")

            # Save to file if requested
            if args.output and result:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"\nResult saved to {args.output}")
        except Exception as e:
            print(f"Error running inference: {e}")
    else:
        print("Error: No input provided. Provide a prompt or file, or use --list-models, --list-providers.")


if __name__ == "__main__":
    main()
