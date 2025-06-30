.. _installation:
.. index:: Installation

Installation
============

Linux
-----

Ideally, you would have installed Blender from a zip-file, and not using a package manager or snap.

Create a new python environment (which we will call render_env here) with a compatible python version and activate it.

    To be able to render isosurfaces, install scikit-image in the environment:
    
    .. code-block:: console
    
       (render_env)$ pip install scikit-image

    If you do not do this, you might get an error at the final step of the install script, however
    the error should be that Blender was not able to import ``skimage``, and because we import :program:`render_molecules`
    first, that should be imported successfully.

Then, while still in that environment, run the ``install.sh`` script:

.. code-block:: console
   
    (render_env)$ ./install.sh 

This will create symlinks to the :program:`render_molecules` directory and ``site-packages/skimage`` directory of the current environment.
Depending on where you installed Blender (for example in ``/usr/local/``), you might need root permissions to make the symlinks.
If so, run the script as

.. code-block:: console

    (render_env)$ sudo -N env "PATH=$PATH" ./install.sh

This will run the script using the non-root-user environment, such that it can still find the Blender executable. If you have not registered Blender
(I would first recommend doing so, see `here <https://docs.blender.org/manual/en/latest/editors/preferences/system.html#prefs-system-register>`_),
you can also pass the path to the executable to the install script, as

.. code-block:: console

    (render_env)$ ./install.sh  /path/to/blender/executable

.. code-block:: console

    (render_env)$ sudo -N env "PATH=$PATH" ./install.sh /path/to/blender/executable

After making the symlinks, the script will try to import :program:`render_molecules` and ``skimage`` in a Blender instance running in the background.

MacOS
-----

The above may work, not sure. I cannot test it myself.

Windows
-------

No idea at the moment.

Uninstalling
------------

Linux
`````

Run ``uninstall.sh``. It has the same requirements as ``install.sh`` discussed above.
