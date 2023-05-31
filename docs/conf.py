# Configuration file for the Sphinx documentation builder.


import os
import sys


# -- Project information

project = 'DQM'
copyright = '2023, Zander Teller'
author = 'Zander Teller'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_toolbox.collapse',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_logo = 'images/dqm_logo.png'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Local build configuration

# are we building on ReadTheDocs or locally?
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    # RTD installs our package via pip (see our requirements.txt file)
    # here, locally, we just add it to the path
    sys.path.insert(0, '.')
# end if building locally
