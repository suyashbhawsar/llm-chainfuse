#!/usr/bin/env python3
import argparse
import json
import logging
import os
from llm_inference import LLMInference
from model_providers import get_provider

def main():
    parser = argparse.ArgumentParser(description="Multi-provider LLM Inference Tool")
    
    # Provider configuration
    provider_group = parser.add_argument_group("Provider Configuration")
    provider_group.add_argument("--provider", type=str, default="openai", 
                             help="LLM provider (openai, anthropic, ollama)")
    provider_group.add_argument("--api-key", type=str, 
                             help="API key for the selected provider (defaults to environment variable)")
    provider_group.add_argument("--provider-options", type=str, 
                             help="JSON string of additional provider options")
    
    # Main arguments
    parser.add_argument("file", nargs="?", type=str, 
                      help="Path to input file (YAML/JSON)")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                      help="Default model for the selected provider")
    parser.add_argument("--temperature", type=float, default=0.7, 
                      help="Default temperature")
    parser.add_argument("--max_tokens", type=int, default=256, 
                      help="Default max tokens")
    parser.add_argument("--top_p", type=float, default=1.0, 
                      help="Default Top-P sampling")
    # Optional OpenAI-specific parameters
    openai_group = parser.add_argument_group("OpenAI-specific Parameters")
    openai_group.add_argument("--seed", type=int, default=None, 
                           help="Seed for reproducibility (OpenAI only)")
    openai_group.add_argument("--frequency_penalty", type=float, default=None, 
                           help="Frequency penalty (OpenAI only)")
    openai_group.add_argument("--presence_penalty", type=float, default=None, 
                           help="Presence penalty (OpenAI only)")
    openai_group.add_argument("--logprobs", type=int, default=None, 
                          help="Log probabilities (OpenAI only)")
    
    parser.add_argument("--output", type=str, 
                      help="Path to output file")
    parser.add_argument("--print", nargs="*", 
                      help="Print results to console. Use without arguments to print all results, or specify prompt IDs to print specific results.")
    
    # Context files
    parser.add_argument("--context", nargs="*", 
                      help="List of context files (ID:path). Example: id1:context1.txt id2:context2.txt")
    
    # Utility functions
    utility_group = parser.add_argument_group("Utility Functions")
    utility_group.add_argument("--list-models", action="store_true", 
                            help="List available models for the selected provider")
    utility_group.add_argument("--list-providers", action="store_true", 
                            help="List available providers")
    utility_group.add_argument("--validate-models", action="store_true", 
                            help="Validate models & parameters in the prompt file")
    
    args = parser.parse_args()
    
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
                
            if env_var and os.environ.get(env_var):
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
        if not args.file:
            print("Error: Please provide a YAML/JSON file for model validation.")
            return
        llm.validate_models(args.file)
        return
    
    # Handle YAML/JSON prompt execution
    if args.file:
        try:
            results = llm.run(args.file, args.output)
            
            # Handle printing results if requested
            if hasattr(args, 'print') and args.print is not None:
                print("\n=== LLM INFERENCE RESULTS ===\n")
                
                # If no specific IDs were provided, print all results
                if len(args.print) == 0:
                    for prompt_id, result in results.items():
                        print(f"\n== RESULT: {prompt_id} ==\n")
                        print(result)
                        print("\n" + "="*50 + "\n")
                else:
                    # Print only the specified prompt IDs
                    for prompt_id in args.print:
                        if prompt_id in results:
                            print(f"\n== RESULT: {prompt_id} ==\n")
                            print(results[prompt_id])
                            print("\n" + "="*50 + "\n")
                        else:
                            print(f"Warning: No result found for prompt ID '{prompt_id}'")
        except Exception as e:
            print(f"Error running inference: {e}")
    else:
        print("Error: No input file provided. Use --list-models, --list-providers, or specify a YAML/JSON file.")


if __name__ == "__main__":
    main()
