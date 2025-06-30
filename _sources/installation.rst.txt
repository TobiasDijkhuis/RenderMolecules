.. _installation:
.. index:: Installation

Installation
============

Linux
-----

Ideally, you would have installed Blender from a zip-file, and not using a package manager or snap.

Create a new python environment (which we will call ``render_env`` here) with a compatible python version and activate it.

    To be able to render isosurfaces, install ``scikit-image`` in the environment:
    
    .. code-block:: console
    
       (render_env)$ pip install scikit-image

    If you do not do this, you might get an error at the final step of the install script, however
    the error should be that Blender was not able to import ``skimage``, and because we import :program:`render_molecules`
    first, that should be imported successfully.

Then, while still in that environment, run the ``install_linux.sh`` script:

.. code-block:: console
   
    (render_env)$ ./install_linux.sh 

This will create symlinks to the :program:`render_molecules` directory and ``site-packages/skimage`` directory of the current environment.
Depending on where you installed Blender (for example in ``/usr/local/``), you might need root permissions to make the symlinks.
If so, run the script as

.. code-block:: console

    (render_env)$ sudo -N env "PATH=$PATH" ./install_linux.sh

This will run the script using the non-root-user environment, such that it can still find the Blender executable. If you have not registered Blender
(I would first recommend doing so, see `here <https://docs.blender.org/manual/en/latest/editors/preferences/system.html#prefs-system-register>`__),
you can also pass the path to the executable to the install script, as

.. code-block:: console

    (render_env)$ ./install_linux.sh  /path/to/blender/executable

or

.. code-block:: console

    (render_env)$ sudo -N env "PATH=$PATH" ./install_linux.sh /path/to/blender/executable

After making the symlinks, the script will try to import :program:`render_molecules` and ``skimage`` in a Blender instance running in the background.

MacOS
-----

Create a new python environment (which we will call ``render_env`` here) with a compatible python version and activate it.

    To be able to render isosurfaces, install ``scikit-image`` in the environment:
    
    .. code-block:: console
    
       (render_env)$ pip install scikit-image

    If you do not do this, you might get an error at the final step of the install script, however
    the error should be that Blender was not able to import ``skimage``, and because we import :program:`render_molecules`
    first, that should be imported successfully.

Then, while still in that environment, run the ``install_macos.sh`` script:

.. code-block:: console
   
    (render_env)$ sh ./install_macos.sh 

This will create symlinks to the :program:`render_molecules` directory and ``site-packages/skimage`` directory of the current environment.
Depending on where you installed Blender (for example in ``/usr/local/``), you might need root permissions to make the symlinks.
If so, run the script as

.. code-block:: console

    (render_env)$ sh ./install_macos.sh

This will run the script using the non-root-user environment, such that it can still find the Blender executable. If you have not registered Blender
(I would first recommend doing so, see `here <https://docs.blender.org/manual/en/latest/editors/preferences/system.html#prefs-system-register>`__),
you can also pass the path to the executable (most likely ``/Applications/Blender.app/Contents/MacOS/Blender``) to the install script, as

.. code-block:: console

    (render_env)$ sh ./install_macos.sh  /path/to/blender/executable

or

.. code-block:: console

    (render_env)$ sudo -N env "PATH=$PATH" sh ./install_macos.sh /path/to/blender/executable

After making the symlinks, the script will try to import :program:`render_molecules` and ``skimage`` in a Blender instance running in the background.

.. note::
    When trying to import ``skimage``, you might get an error with::

       (mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64'))

    If that is the case, your Mac is likely on Apple Silicon, and not on an Intel chip. To make sure your environment is compatible with this,
    please remove the environment, and recreate it as

    .. code-block:: console

        (render_env)$ conda deactivate
        (base)$ conda remove -n render_env --all
        (base)$ CONDA_SUBDIR=osx-arm64 conda create -n render_env python=***
        (base)$ conda activate render_env
        (render_env)$ pip cache purge
        (render_env)$ pip install scikit-image

    This will ensure that newly installed packages are compatible with the ARM architecture.
    For more information, see `here <https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple>`__.

Windows
-------

No idea at the moment.

Uninstalling
------------

Linux
`````

Run ``uninstall.sh``. It has the same requirements as ``install_linux.sh`` discussed above.
