..

Installation
============

Install a Python Virtual Env
----------------------------

We recommend to use Ubuntu 20.04. Create a workspace for storing clt packages and the virtual environment:

.. code-block:: bash

    mkdir robot-clutter
    cd robot-clutter

    sudo apt-get install python3-tk python3-pip
    sudo pip3 install virtualenv
    virtualenv env --python=python3 --prompt='[clutter-env] '
    echo "export ROBOT_CLUTTER_WS=$PWD" >> env/bin/activate

Then activate the environment:

.. code-block:: bash

     source env/bin/activate

Clone and install core and assets:

.. code-block:: bash

    git clone https://github.com/robot-clutter/clt_assets.git
    cd clt_assets
    pip install -e .
    cd ..

    git clone https://github.com/robot-clutter/clt_core.git
    cd clt_core
    pip install -e .
    cd ..

Finally, install pytorch:

.. code-block:: bash

    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
d

Generate documentation locally
------------------------------

If you want to generate the documentation locally run the following:

.. code-block:: bash

    cd $ROBOT_CLUTTER_WS/clt_core/doc
    make html

Then, open `doc/_build/html/index.html` on your browser.

.. toctree::
