import os
from pathlib import Path
from sphinx.application import Sphinx
from sphinx.util.docutils import docutils_namespace

DOC_PATH = Path("doc/next")
TARGET_PATH = Path("processed")
BUILD_PATH = Path("_build")

def main():
    TARGET_PATH.mkdir(parents=True, exist_ok=True)
    BUILD_PATH.mkdir(parents=True, exist_ok=True)
    
    app = Sphinx(
        srcdir=str(DOC_PATH),
        confdir=str(DOC_PATH),
        outdir=str(BUILD_PATH / "markdown"),
        doctreedir=str(BUILD_PATH / "doctree"),
        buildername="markdown",
        confoverrides={
            'master_doc': 'index',
            'source_suffix': '.md',
            'extensions': [
                'myst_parser',
                'sphinx_copybutton',
                'sphinx_design',
            ],
            'myst_enable_extensions': [
                'colon_fence',
                'deflist',
                'dollarmath',
                'fieldlist',
                'html_admonition',
                'html_image',
                'replacements',
                'smartquotes',
                'tasklist',
            ],
            'myst_heading_anchors': 3,
            'copybutton_prompt_text': '$ ',
        },
        freshenv=True,
    )

    with docutils_namespace():
        app.build()

    import shutil
    for md_file in (BUILD_PATH / "markdown").rglob("*.md"):
        relative_path = md_file.relative_to(BUILD_PATH / "markdown")
        target_file = TARGET_PATH / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, target_file)

if __name__ == "__main__":
    main() 