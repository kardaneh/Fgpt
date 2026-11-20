Derived Types
=============

FGPT translates every Fortran derived type into a Python ``@dataclass`` that
**always inherits from** ``FortranDerivedType`` (see :doc:`fortran_io_runtime`)
and is registered as a JAX pytree. This holds for simple types, arrays of
types and nested types. Storage is always *structure of arrays* (SoA): there
is no list-of-objects form. What changes between uses is how an instance is
built and which helpers apply:

* **Simple derived types** (one instance) are built by calling the class;
  ``_shape`` is ``()``.
* **Arrays of derived types** are built with ``zeros(shape)``, which puts the
  array dimensions first on every field.
* **Nested derived types** (a component that is itself a derived type)
  compose the two cases.

The type of a field (scalar or array-valued) does not change this.

.. contents:: On this page
   :local:
   :depth: 2

Simple Derived Types
--------------------

.. code-block:: fortran

   type point_type
     real :: x
     real :: y(3)
   end type

   type(point_type) :: p

.. code-block:: python

   @dataclass
   class point_type(FortranDerivedType):
       x: np.float64 = np.float64(0)
       y: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))

   jax.tree_util.register_dataclass(point_type)

   p = point_type()      # p._shape == ()

Having an array-valued field such as ``y`` does **not** make the type an
array of derived types.

Writes in JAX rebind the struct with ``with_field``:

.. code-block:: python

   # NumPy
   p.x = 1.0
   p.y[1] = 2.0
   # JAX
   p = p.with_field('x', 1.0)
   p = p.with_field('y', p.y.at[1].set(2.0))

Arrays of Derived Types
-----------------------

An array of derived types is one object whose fields are arrays. Outer
(array) dimensions come first, followed by the field's own dimensions.

.. code-block:: fortran

   type(laieff_type) :: laieff_fit(kjpindex, nvm)

.. code-block:: python

   laieff_fit = laieff_type.zeros((kjpindex, nvm))
   # laieff_fit.a.shape == (kjpindex, nvm)
   # laieff_fit.b.shape == (kjpindex, nvm, 10)
   # laieff_fit._shape  == (kjpindex, nvm)

Element access is flattened, outer indices first:

.. code-block:: python

   laieff_fit(i, j)%a      ->  laieff_fit.a[i, j]
   laieff_fit(i, j)%b(k)   ->  laieff_fit.b[i, j, k]

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - NumPy (after transpilation)
     - JAX (after conversion)
   * - Created by
     - ``<type>.zeros(shape)``
     - ``jax.tree_util.tree_map(jnp.asarray, ...)``
   * - Mutability
     - Mutable: ``x.a[i] = value`` works
     - Immutable: use ``with_field`` / ``set``
   * - ``_shape``
     - Read-only property, derived from the leaves
     - Same property; preserved by ``tree_map``
   * - Element update
     - ``__setitem__``
     - ``x.set(i, value)``

``_shape`` is a property, not a stored attribute, so it is recomputed from
the leaves and stays correct after indexing and ``tree_map``.

.. note::

   ``x[i]`` returns views for basic indices and copies for fancy indices
   (lists, arrays, masks). Chained writes such as ``m[[0, 2]].a[0] = 1.0``
   would modify a temporary and are lost, so the transpiler never emits
   them. Reads and by-value passing may use ``x[i]``; **every write is one
   flat subscript on the leaf**, e.g. ``x.a[i, j]``.

Nested Derived Types
--------------------

.. code-block:: fortran

   type Component
     real(8) :: a(3)
   end type

   type Model
     type(Component) :: one
     type(Component) :: items(5)
   end type

.. code-block:: python

   @dataclass
   class Component(FortranDerivedType):
       a: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))

   @dataclass
   class Model(FortranDerivedType):
       one: Component = field(default_factory=Component)
       items: Component = field(default_factory=lambda: Component.zeros((5,)))

* A scalar nested component uses ``field(default_factory=Inner)``. A bare
  ``Inner()`` default is rejected by dataclasses as a mutable default.
* An array of nested components is annotated with the nested type (never
  ``list[...]``) and built with ``Inner.zeros(shape)``.
* ``Model.zeros((n,))`` expands recursively: ``items.a`` has shape
  ``(n, 5, 3)`` and ``one.a`` has shape ``(n, 3)``.

Access chains are flattened: indices from every struct level are collected,
outer first, and applied once on the leaf.

.. code-block:: python

   laieff_fit%comp%temp(2)       ->  laieff_fit.comp.temp[2]
   model%items(i)%a(k)           ->  model.items.a[i, k]
   model(i)%items(j)%a(k)        ->  model.items.a[i, j, k]
   laieff_fit(i,j,k)%a(i)        ->  laieff_fit.a[i, j, k, i]

Index adjustment treats ``x.items.a[...]`` as one array whose rank is the
outer dimensions plus every array level along the chain.

Class metadata
~~~~~~~~~~~~~~

Dataclass entries in the class metadata (``cls_info``) have
``is_dataclass: True`` and one dict per field:

* scalar: ``{"kind": "scalar", "dtype": ...}``
* array: ``{"kind": "array", "dtype": ..., "shape": [...], "lower_bounds": [...]}``
* nested: ``{"kind": "dataclass", "type": ..., "is_array": bool, "shape": ...}``

Derived-type attributes of the global module use the same dict format.

JAX Registration
----------------

Every derived-type dataclass is registered with
``jax.tree_util.register_dataclass``, **inner types before outer types**.
They work as pytrees under ``jit``, ``scan``, ``cond``, ``vmap`` and
differentiation. ``_shape`` is a property, so registration ignores it.

Translation Rules (JAX)
-----------------------

Field writes
~~~~~~~~~~~~

.. code-block:: python

   # NumPy
   laieff_fit.a[1, 0, 1] = 42.0
   # JAX
   laieff_fit = laieff_fit.with_field('a', laieff_fit.a.at[1, 0, 1].set(42.0))

   # nested (verify against converter output)
   model = model.with_field('items',
       model.items.with_field('a', model.items.a.at[1, 2].set(7.0)))

Accumulations use ``.at[...].add`` or ``.at[...].multiply``. Slices are
preserved. The original object is never modified.

.. warning::

   ``x.a.at[...].set(...)`` without rebinding the result is a no-op. Always
   use the generated ``with_field`` form.

Vectorized loops
~~~~~~~~~~~~~~~~

Inside a vectorized loop, the batch index is replaced by a slice, for reads
and writes:

.. code-block:: python

   laieff_fit.a[ji, ivm, ilevel] = v[ji]
   # becomes
   laieff_fit = laieff_fit.with_field(
       'a', laieff_fit.a.at[:, ivm, ilevel].set(v))

   calculate_laieff_fit(angle, laieff_fit[ji, ivm, ilevel])
   # becomes
   calculate_laieff_fit(angle, laieff_fit[:, ivm, ilevel])

A struct field whose name matches a lifted local variable does not collide
with it.

Masked writes
~~~~~~~~~~~~~

Conditional writes in vectorized code become ``jnp.where``. For array-valued
fields the mask is expanded (for example ``_mask[:, None]``) to match the
field rank. Scalar fields are not expanded. Accumulations use a per-field
delta variable so two fields under the same mask do not collide.

Structs on ``self``
~~~~~~~~~~~~~~~~~~~

For ``self.soil_fit.a[0] = 1.0``, the first write reads ``self.soil_fit`` and
later reads and writes use the local ``soil_fit``. The struct is registered
as a modified attribute and the updated module is returned with
``eqx.tree_at``.

Structs as arguments
~~~~~~~~~~~~~~~~~~~~

A struct passed to a function and written inside it is registered as a
modified argument.

Dimension inference
~~~~~~~~~~~~~~~~~~~

Declared dimensions of a field are the struct's outer dimensions plus the
field's inner dimensions and, for nested types, the dimensions of each array
level along the chain. Indexing with a scalar removes the corresponding
active dimension.

Numerical Behavior
------------------

The test suite checks that:

* ``float64`` fields stay ``float64`` in ``scan`` carries.
* Updated structs can be returned from ``jit`` and carried through ``scan``
  and ``cond`` (both branches must have the same structure).
* ``jax.grad``, ``jax.jvp`` and ``jax.vjp`` work through field writes and
  through ``x[ji, ...]`` under ``vmap``.
* ``_shape`` survives ``tree_map``.

Limitations
-----------

* Deferred-shape components (``ALLOCATABLE`` or ``(:)`` inside a derived
  type) raise ``NotImplementedError``.
* Pointer components and polymorphic types (``CLASS``) are not supported yet.
* Initializers on arrays of derived types are not supported.
* Non-default lower bounds on components are recorded in the metadata, but
  index adjustment for them is still being completed.
* ``x[i] = value`` fails on the JAX object. The converter emits
  ``with_field`` or ``set`` instead.
