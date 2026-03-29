from importlib.metadata import metadata

# Project information
dist = metadata("sdiff")
project = dist.get("name")
author = dist.get("author-email")
version = release = dist.get("version")
copyright = (
    f"{dist.get('author-email')} and contributors. All rights reserved."
    f" Distributed under the terms of the"
    f" {dist.get('license')} license. See LICENSE.md for more details."
)

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
