from .ssmodels.base_model import Registries


class ModelZoo:
    
    def __init__(self, key, module_path, use_gpu, kwargs):
        """
        Select a model from the model zoo according to the key. Validate the choice and load the relevant model.
        
        Args:
            key (str): The name of the model to be loaded, must be the same as when registering the model.
            module_path (str): The path to the module containing the model code.
            use_gpu (bool): Whether to use the GPU or not.
            **kwargs: Additional arguments to be passed to the model.
            
        Returns:
            ModelZoo: The instantiated model instance."""
        Registries.ModelChoice.validate_choice(key) 
        selected_single_step_model = Registries.ModelChoice.get_choice(key)
        self.model_instance = selected_single_step_model(module_path)
        self.model_instance.model_setup(use_gpu, **kwargs)
        
    def model_call(self, input):
        """
        Call the model instance with the input.

        Args:
            input (list): List of SMILES strings.

        Returns:
            
        """
        return self.model_instance.model_call(input)

    