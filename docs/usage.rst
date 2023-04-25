Usage
=====

.. _installation:

Installation
------------

To use DQM, **don't** install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``dqm.DQM.run_simple()`` method:

.. autofunction:: dqm.DQM.run_simple

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`dqm.DQM.run_simple`
will raise an exception.

# .. autoexception:: lumache.InvalidKindError

For example:

>>> import dqm
>>> dqm.DQM.run_simple()
['shells', 'gorgonzola', 'parsley']

