import argparse


class FileTypeWithExtensionCheck(argparse.FileType):
    """
    Custom `argparse.FileType` with additional extension validation.

    ---
    ### Parameters:
        - `mode` (`str`, optional): File mode. Default is `r`.
        - `valid_extensions` (`str` or `Tuple[str, ...]`, optional): Valid file extensions.
        - `**kwargs`: Additional keyword arguments.

    This class extends argparse.FileType to include validation of file extensions.
    """

    def __init__(self, mode="r", valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        """
        Validate the file extension before calling the parent `__call__` method.

        ---
        ### Parameters:
            - `string` (`str`): Input string representing the filename.

        ---
        ### Returns:
            - `file`: File object.

        This method performs additional validation on the file extension before calling
        the parent `__call__` method from `argparse.FileType`.
        """
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError("Not a valid filename extension!")
        return super().__call__(string)


class FileTypeWithExtensionCheckWithPredefinedDatasets(FileTypeWithExtensionCheck):
    """
    Custom `argparse.FileType` with additional extension and predefined dataset validation.

    ---
    ### Parameters:
        - `mode` (`str`, optional): File mode. Default is `r`.
        - `valid_extensions` (`str` or `Tuple[str, ...]`, optional): Valid file extensions.
        - `available_datasets` (`List[str]`, optional): List of available datasets.
        - `**kwargs`: Additional keyword arguments.

    This class extends `FileTypeWithExtensionCheck` to include validation of predefined datasets.
    """

    def __init__(
        self, mode="r", valid_extensions=None, available_datasets=None, **kwargs
    ):
        super().__init__(mode, valid_extensions, **kwargs)
        self.available_datasets = available_datasets or []

    def __call__(self, string):
        """
        Validate the file extension and predefined dataset before calling the parent `__call__` method.

        ---
        ### Parameters:
            - `string` (`str`): Input string representing the filename.

        ---
        ### Returns:
            - `file` or `str`: File object or predefined dataset name.

        This method performs additional validation on the file extension and predefined dataset before calling
        the parent `__call__` method from `FileTypeWithExtensionCheck`.
        """
        if len(self.available_datasets) > 0 and string in self.available_datasets:
            return string
        return super().__call__(string)
