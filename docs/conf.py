from importlib.metadata import version

project = "DETECTIV"
author = "Mateusz Smendowski"
release = version("detectiv")
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

html_theme = "pydata_sphinx_theme"
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build"]
nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
