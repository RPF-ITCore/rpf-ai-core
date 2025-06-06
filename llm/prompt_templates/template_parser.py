import os
import logging
import importlib
from string import Template

class TemplateParser:
    """
    Loads and renders prompt templates from locale-specific modules.
    """
    def __init__(self, language: str = None, default_language: str = "en"):
        # Base directory containing locales folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.locales_dir = os.path.join(base_dir, "locales")
        self.default_language = default_language
        self.language = default_language
        self.logger = logging.getLogger(__name__)
        # Attempt to set requested language if provided
        if language:
            self.set_language(language)

    def set_language(self, language: str):
        """
        Set active locale if available; otherwise fallback to default.
        """
        path = os.path.join(self.locales_dir, language)
        if language and os.path.isdir(path):
            self.language = language
            self.logger.info(f"Language set to: {language}")
        else:
            self.logger.warning(
                f"Locale '{language}' not found, defaulting to '{self.default_language}'"
            )
            self.language = self.default_language

    def get_template(self, group: str, key: str, vars: dict = None) -> str:
        """
        Retrieve and render a named template.

        :param group: Name of the prompt group file (without .py extension)
        :param key:   Identifier of the template within the group
        :param vars:  Mapping of placeholder names to values
        :return:      Rendered prompt string
        """
        vars = vars or {}
        if not group or not key:
            self.logger.error("Both 'group' and 'key' must be specified.")
            return ""

        # Try primary locale, then fallback to default
        for locale in (self.language, self.default_language):
            module_path = f"llm.prompt_templates.locales.{locale}.{group}"
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                self.logger.debug(f"Cannot import {module_path}: {e}")
                continue

            # If module defines a PROMPTS dict, use it
            if hasattr(module, 'PROMPTS') and isinstance(module.PROMPTS, dict):
                prompts = module.PROMPTS
                if key in prompts:
                    template_obj = prompts[key]
                else:
                    self.logger.error(
                        f"Key '{key}' not found in PROMPTS of {module_path}"
                    )
                    return ""
            else:
                # Otherwise, expect a top-level variable named key
                if hasattr(module, key):
                    template_obj = getattr(module, key)
                else:
                    self.logger.error(
                        f"Key '{key}' not found in module {module_path}"
                    )
                    return ""

            # Render Template or format string
            if isinstance(template_obj, Template):
                return template_obj.substitute(vars)
            if isinstance(template_obj, str):
                try:
                    return template_obj.format(**vars)
                except Exception as e:
                    self.logger.error(f"Error formatting string template: {e}")
                    return template_obj

            self.logger.error(
                f"Unsupported template type for {module_path}.{key}: {type(template_obj)}"
            )
            return str(template_obj)

        # If we reach here, none of the locales loaded
        self.logger.error(
            f"Failed to load template '{group}.{key}' in any locale."
        )
        return ""
