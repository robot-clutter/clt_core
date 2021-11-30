..

Installation
============

Install a Python Virtual Env
----------------------------

We recommend to use Ubuntu 20.04. Then install `python3-tk` and `virtualenv`:

.. code-block:: bash

    sudo apt-get install python3-tk

.. code-block:: bash

    sudo pip3 install virtualenv
    virtualenv ~/clutter-env --python=python3 --prompt='[clutter-env] '

At the end of the ~/clutter-env/bin/activate` script, add the following lines:

.. code-block:: bash

    # If the virtualenv inherits the `$PYTHONPATH` from your system:
    export PYTHONPATH="$VIRTUAL_ENV/lib"

Then, activate the environment (you have to activate it each time you want to use it):

.. code-block:: bash

 source ~/clutter-env/bin/activate

Install Clutter
---------------
Clone the repository and install the package:

.. code-block:: bash

   git clone https://github.com/robot-clutter/clutter.git
   cd clutter
   pip install -e .


Generate documentation locally
------------------------------

If you want to generate the documentation locally run the following:

.. code-block:: bash

    cd doc
    make html

Then, open `doc/_build/html/index.html` on your browser.

.. toctree::
