Installation
============

GenSec is written in Python 3 and is currently used from source.

Requirements
++++++++++++

- Python 3
- Git
- ASE-compatible calculator backend for geometry optimization (for example, an ASE calculator script configured in your input file)

Core dependencies
+++++++++++++++++

Install the core Python dependencies:

.. code-block:: sh

    pip install numpy ase scipy networkx timeout-decorator

Optional dependencies
++++++++++++++++++++

Install these only if you use FPS/descriptor-based structure selection:

.. code-block:: sh

    pip install featomic metatensor scikit-matter

Install from source
+++++++++++++++++++

Clone the repository and move into it:

.. code-block:: sh

    git clone https://github.com/sabia-group/gensec.git
    cd gensec

Create and activate a virtual environment (recommended):

.. code-block:: sh

    python3 -m venv .venv
    source .venv/bin/activate

Add the repository root to ``PYTHONPATH``:

.. code-block:: sh

    export PYTHONPATH=$PYTHONPATH:$PWD

Test 
++++++++++

From the repository root, run:

.. code-block:: sh

    python gensec.py inputs/parameters_generate.json

If the run starts and creates output folders/databases, GenSec is installed correctly.
