..

Conventions
===========

Here we list some conventions we use in the code to avoid unnecessary inconsistencies.

Code documentation style
------------------------

We use the code documentation style that numpy uses. More info `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Images
------

The reference frame in the image plane is placed with the `x` axis from left to right (width of the image) and the `y` axis from top to bottom (along the height of the image).

Every variable which represents the position of a pixel on an image contains the coordinates with respect to this frame, i.e.:

.. code-block:: python

    pos = [x, y]

This means that if we have a numpy array storing the image, we retrieve the value of a pixel as follows:

.. code-block:: python

    value = image[y, x]  # or
    value = image[pos[1], pos[0]]

because the rows of the array correspond to the height (y dimension) and the columns to the width (x dimension).

If a vector represents the size of the image, then we have

.. code-block:: python

    size = (size_along_x, size_along_y)

which is different from the size of an numpy array by getting its shape, because the shape of a matrix is `(nr_rows, nr_cols)`. This means that if we read the size of an image by the shape of the array (`image.shape`) we have to flip the elements to represent the size of an image correctly, e.g.:

.. code-block:: python

    image_size = image.shape[::-1]


Inertial frames
---------------
The inertia frame is the frame of the table. Inputs and outputs of poses assume that are expressed with respect the inertia  frame. The inertia frame is on the surface of the table not the center of its geometry.

Cartesian poses
---------------

We represent a Cartesian pose as the pair `(pos, quat)`, where pos is always an np.array of size 3 and quat is an object of the class Quaternion(). Notice, that PyBullet represents quaternions as a list with 4 elements in the order `xyzw`. In order to avoid confusion for the order of the quaternion elements, we use the class Quaternion() wherever possible. This class, also, contains tools for different representation of the orientation (rotation matrix etc).

Size
----
The name `size` refers to the distance from the center of the geometry to its edge. For example, the size of a sphere is its radius. The size of rectangle is three numbers which are half its length, width, height.

.. toctree::
