#!/usr/bin/env python3
"""
Environment loader with support for both static values and executable commands.

This module extends standard .env functionality to support executing bash commands
for retrieving sensitive data like API keys from external sources (e.g., 1Password CLI).

Example .env file:
    # Static values
    PROVIDER=openai
    MODEL=gpt-4o
    
    # Executable commands (prefixed with 'exec:')
    OPENAI_API_KEY=exec:op read "op://DevOps Dynamics/ngk2owbnrflxpqsbtrttoio3se/credential"
    ANTHROPIC_API_KEY=exec:security find-internet-password -s anthropic.com -a myuser -w
    
    # Mixed usage
    DEBUG=true
    GOOGLE_API_KEY=exec:echo $GOOGLE_TOKEN
"""

import os
import subprocess
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

class EnvLoader:
    """Enhanced environment loader with command execution support."""
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize the environment loader.
        
        Args:
            env_file_path: Path to .env file. If None, searches for .env in current and parent dirs.
        """
        self.env_file_path = env_file_path or find_dotenv()
        self.loaded_vars: Dict[str, str] = {}
        
    def load(self, override: bool = True) -> Dict[str, str]:
        """
        Load environment variables from .env file, executing commands as needed.
        
        Args:
            override: Whether to override existing environment variables
            
        Returns:
            Dictionary of loaded environment variables
        """
        if not self.env_file_path or not os.path.exists(self.env_file_path):
            logger.info(f"No .env file found at {self.env_file_path or '.env'}")
            return {}
        
        # First load the .env file normally to get all variables
        load_dotenv(self.env_file_path, override=override)
        
        # Read the file manually to process executable commands
        try:
            with open(self.env_file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            return {}
        
        processed_vars = {}
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Parse key=value pairs
            if '=' not in line:
                continue
                
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            # Check if this is an executable command
            if value.startswith('exec:'):
                command = value[5:].strip()  # Remove 'exec:' prefix
                try:
                    result = self._execute_command(command)
                    processed_vars[key] = result
                    
                    # Set in environment if override is True or var doesn't exist
                    if override or key not in os.environ:
                        os.environ[key] = result
                        
                    logger.debug(f"Executed command for {key}: {command[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to execute command for {key} at line {line_num}: {e}")
                    logger.error(f"Command: {command}")
                    continue
            else:
                # Regular static value
                processed_vars[key] = value
                
        self.loaded_vars.update(processed_vars)
        logger.info(f"Loaded {len(processed_vars)} environment variables from {self.env_file_path}")
        
        return processed_vars
    
    def _execute_command(self, command: str) -> str:
        """
        Execute a shell command and return its output.
        
        Args:
            command: Shell command to execute
            
        Returns:
            Command output stripped of whitespace
            
        Raises:
            subprocess.CalledProcessError: If command fails
            Exception: For other execution errors
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                check=True
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out after 30 seconds: {command}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Command failed with exit code {e.returncode}: {e.stderr.strip()}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.environ.get(key, default)
    
    def get_loaded_vars(self) -> Dict[str, str]:
        """
        Get all variables that were loaded from the .env file.
        
        Returns:
            Dictionary of loaded variables
        """
        return self.loaded_vars.copy()
    
    def create_example_env(self, file_path: str = ".env.example") -> None:
        """
        Create an example .env file with documentation.
        
        Args:
            file_path: Path where to create the example file
        """
        example_content = '''# LLM ChainFuse Environment Configuration
# ==========================================

# API Keys - Static values
# ------------------------
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# API Keys - Using executable commands (prefix with 'exec:')
# -----------------------------------------------------------
# 1Password CLI examples:
# OPENAI_API_KEY=exec:op read "op://DevOps Dynamics/openai-key/credential"
# ANTHROPIC_API_KEY=exec:op read "op://DevOps Dynamics/anthropic-key/credential"
# GOOGLE_API_KEY=exec:op read "op://DevOps Dynamics/google-key/credential"

# macOS Keychain examples:
# OPENAI_API_KEY=exec:security find-internet-password -s openai.com -a your-username -w
# GOOGLE_API_KEY=exec:security find-generic-password -s google-api -a your-username -w

# Environment variable examples:
# OPENAI_API_KEY=exec:echo $OPENAI_TOKEN
# GOOGLE_API_KEY=exec:cat /path/to/secret/file

# AWS CLI / kubectl examples:
# AWS_ACCESS_KEY=exec:aws configure get aws_access_key_id
# KUBE_TOKEN=exec:kubectl get secret mytoken --template='{{.data.token}}' | base64 -d

# Bitwarden CLI examples:
# OPENAI_API_KEY=exec:bw get password "OpenAI API Key"
# ANTHROPIC_API_KEY=exec:bw get notes "Anthropic Credentials" | jq -r '.api_key'

# Other sensitive configuration (optional)
# ----------------------------------------
# DATABASE_URL=exec:op read "op://DevOps Dynamics/database-url/credential"
# REDIS_PASSWORD=exec:security find-generic-password -s redis -a myapp -w
'''
        
        try:
            with open(file_path, 'w') as f:
                f.write(example_content)
            logger.info(f"Created example .env file at {file_path}")
        except Exception as e:
            logger.error(f"Failed to create example .env file: {e}")


# Global instance for easy importing
env_loader = EnvLoader()

def load_env(env_file_path: Optional[str] = None, override: bool = True) -> Dict[str, str]:
    """
    Convenience function to load environment variables.
    
    Args:
        env_file_path: Path to .env file
        override: Whether to override existing environment variables
        
    Returns:
        Dictionary of loaded environment variables
    """
    loader = EnvLoader(env_file_path) if env_file_path else env_loader
    return loader.load(override)

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return env_loader.get(key, default) 