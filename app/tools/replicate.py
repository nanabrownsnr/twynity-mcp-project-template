"""Replicate MCP tools for running models and managing resources.

This module provides the core tools for interacting with Replicate's API,
including running models, listing models, getting model information, and checking
connection status.
"""

from typing import Optional, Dict, Any, List
from fastmcp.apps import AppConfig
from fastmcp.tools import ToolResult

from app.tools.connection import get_current_connection
from app.ui.replicate.resource import VIEW_URI


def register_tools(mcp):
    """Register Replicate tools with the FastMCP server."""
    
    @mcp.tool(
        app=AppConfig(
            resource_uri=VIEW_URI,
            visibility=["model", "app"],
        )
    )
    async def run_model(
        model: str,
        input_params: Dict[str, Any],
        version: Optional[str] = None
    ) -> ToolResult:
        """Run a Replicate model with the given parameters.
        
        Args:
            model: The name of the model to run (format: owner/model-name)
            input_params: Input parameters for the model
            version: Optional version of the model to use
            
        Returns:
            A ToolResult with model output and metadata
            
        Examples:
            ``run_model(model="stability-ai/stable-diffusion", 
                      input_params={"prompt": "a beautiful landscape"})``
        """
        # Get connection credentials from database
        connection = await get_current_connection()
        if not connection:
            raise Exception("No Replicate API connection found. Please configure your API key.")
        
        api_token = connection.get("api_key")
        if not api_token:
            raise Exception("Replicate API token not found in connection. Please configure your API key.")
            
        try:
            import replicate
            from replicate.exceptions import ReplicateError
            
            # Configure the client with the user's API token
            replicate_client = replicate.Client(api_token=api_token)
            
            # Build the prediction parameters
            prediction_params = {
                "input": input_params
            }
            
            if version:
                prediction_params["version"] = version
                
            # Run the model prediction
            output = replicate_client.predictions.create(
                model=model,
                **prediction_params
            )
            
            # Wait for completion (this is a blocking operation, so we'll return the pending result)
            # For a real implementation, you might want to use streams or handle async results differently
            
            message = f"Model {model} prediction started successfully. Prediction ID: {output.id}"
            
            return ToolResult(
                content=message,
                structured_content={
                    "status": "success", 
                    "prediction_id": output.id,
                    "model": model,
                    "version": version or "latest",
                    "input_params": input_params,
                    "output": str(output)
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )
            
        except Exception as e:
            error_msg = f"Failed to run model {model}: {str(e)}"
            return ToolResult(
                content=error_msg,
                structured_content={
                    "status": "error",
                    "error": str(e),
                    "model": model
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )
    
    
    @mcp.tool(
        app=AppConfig(
            resource_uri=VIEW_URI,
            visibility=["model", "app"],
        )
    )
    async def list_models() -> ToolResult:
        """List all public models available on Replicate.
        
        Returns:
            A ToolResult with the list of models and their details
            
        Examples:
            ``list_models()``
        """
        # Get connection credentials from database
        connection = await get_current_connection()
        if not connection:
            raise Exception("No Replicate API connection found. Please configure your API key.")
        
        api_token = connection.get("api_key")
        if not api_token:
            raise Exception("Replicate API token not found in connection. Please configure your API key.")
            
        try:
            import replicate
            from replicate.exceptions import ReplicateError
            
            # Configure the client with the user's API token
            replicate_client = replicate.Client(api_token=api_token)
            
            # List models
            models = replicate_client.models.list()
            
            model_list = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "owner": model.owner,
                    "description": getattr(model, 'description', ''),
                    "url": model.url,
                    "latest_version": getattr(model, 'latest_version', {}).get('id', '') if hasattr(model, 'latest_version') else '',
                    "created_at": getattr(model, 'created_at', '')
                }
                model_list.append(model_info)
            
            message = f"Found {len(model_list)} public models on Replicate"
            
            return ToolResult(
                content=message,
                structured_content={
                    "status": "success",
                    "models": model_list
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )
            
        except Exception as e:
            error_msg = f"Failed to list models: {str(e)}"
            return ToolResult(
                content=error_msg,
                structured_content={
                    "status": "error",
                    "error": str(e)
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )


    @mcp.tool(
        app=AppConfig(
            resource_uri=VIEW_URI,
            visibility=["model", "app"],
        )
    )
    async def get_model_info(model: str) -> ToolResult:
        """Get information about a specific Replicate model.
        
        Args:
            model: The name of the model to get info for (format: owner/model-name)
            
        Returns:
            A ToolResult with the model's details
            
        Examples:
            ``get_model_info(model="stability-ai/stable-diffusion")``
        """
        # Get connection credentials from database
        connection = await get_current_connection()
        if not connection:
            raise Exception("No Replicate API connection found. Please configure your API key.")
        
        api_token = connection.get("api_key")
        if not api_token:
            raise Exception("Replicate API token not found in connection. Please configure your API key.")
            
        try:
            import replicate
            from replicate.exceptions import ReplicateError
            
            # Configure the client with the user's API token
            replicate_client = replicate.Client(api_token=api_token)
            
            # Get model info
            model_info = replicate_client.models.get(model)
            
            info = {
                "name": model_info.name,
                "owner": model_info.owner,
                "description": getattr(model_info, 'description', ''),
                "url": model_info.url,
                "latest_version": getattr(model_info, 'latest_version', {}).get('id', '') if hasattr(model_info, 'latest_version') else '',
                "created_at": getattr(model_info, 'created_at', ''),
                "visibility": getattr(model_info, 'visibility', ''),
                "forked_from": getattr(model_info, 'forked_from', None)
            }
            
            message = f"Retrieved information for model {model}"
            
            return ToolResult(
                content=message,
                structured_content={
                    "status": "success",
                    "model": info
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )
            
        except Exception as e:
            error_msg = f"Failed to get model info for {model}: {str(e)}"
            return ToolResult(
                content=error_msg,
                structured_content={
                    "status": "error",
                    "error": str(e),
                    "model": model
                },
                meta={"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI},
            )


    @mcp.tool()
    async def check_replicate_connection() -> ToolResult:
        """Check if the Replicate connection is valid by testing an API call.
        
        Returns:
            A ToolResult with the connection status
            
        Examples:
            ``check_replicate_connection()``
        """
        try:
            # Get connection credentials from database
            connection = await get_current_connection()
            if not connection:
                raise Exception("No Replicate API connection found. Please configure your API key.")
            
            api_token = connection.get("api_key")
            if not api_token:
                raise Exception("Replicate API token not found in connection. Please configure your API key.")
                
            import replicate
            from replicate.exceptions import ReplicateError
            
            # Configure the client with the user's API token and test the connection
            replicate_client = replicate.Client(api_token=api_token)
            
            # Simple API call to validate connection
            _ = replicate_client.models.list()
            
            return ToolResult(
                content="Replicate connection is valid",
                structured_content={
                    "status": "connected",
                    "message": "Successfully connected to Replicate API"
                }
            )
            
        except Exception as e:
            return ToolResult(
                content=f"Replicate connection failed: {str(e)}",
                structured_content={
                    "status": "error",
                    "error": str(e)
                }
            )