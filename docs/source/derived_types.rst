Derived Types
=============

FGPT translates every Fortran derived type into a Python ``@dataclass`` that
**always inherits from** ``FortranDerivedType`` (see :doc:`fortran_io_runtime`)
and is registered as a JAX pytree. This holds for simple types, arrays of
types and nested types alike. The base class exists in both phases of the
pipeline, but its content differs:

* **After the Fortran-to-NumPy transpilation**, it provides ``zeros``,
  ``_shape``, ``__getitem__`` and ``__setitem__``, which operate in place.
* **After the JAX conversion**, it also gains ``with_field`` and ``set``.
  JAX arrays are immutable, so every write is a functional update.

Which helpers apply depends on how the type is used:

* **Simple derived types** (one instance) are built by calling the class and
  written with ``with_field`` in JAX.
* **Arrays of derived types** are stored as a *structure of arrays* (SoA) and
  additionally use ``zeros``, ``_shape``, ``__getitem__``, ``__setitem__``
  and ``set``.
* **Nested derived types** (a component that is itself a derived type) compose
  the two cases above. See `Nested Derived Types`_.

The kind of the fields (scalar or array-valued) does not change this
distinction. Storage is always SoA: there is no list-of-objects form.

.. contents:: On this page
   :local:
   :depth: 2

The Base Class
--------------

**NumPy phase** (output of the transpilation):

.. code-block:: python

   class FortranDerivedType:

       @classmethod
       def zeros(cls, shape):
           return cls._expand(cls(), tuple(shape))

       @staticmethod
       def _expand(default, shape): ...      # recursive, see below

       @property
       def _shape(self): ...                 # derived from the leaves

       def __getitem__(self, idx): ...
       def __setitem__(self, idx, value): ...

**JAX phase** (after the JAX conversion adds two methods):

.. code-block:: python

   class FortranDerivedType:
       # zeros, _shape, __getitem__ and __setitem__ as above, plus:

       def with_field(self, name, value):
           return replace(self, **{name: value})

       def set(self, idx, value):
           return type(self)(**{
               f.name: getattr(self, f.name).at[idx].set(getattr(value, f.name))
               for f in fields(self)
           })

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Member
     - Phase
     - Used by
     - Purpose
   * - ``zeros(shape)``
     - NumPy
     - Arrays of derived types
     - Builds the structure of arrays. ``shape`` becomes a prefix of the
       dimensions of every leaf, at every nesting depth (see ``_expand``).
   * - ``_expand(default, shape)``
     - NumPy
     - ``zeros``
     - Recurses into nested dataclasses and prefixes ``shape`` onto each leaf.
       Without it a nested dataclass default would become an ``object`` array.
   * - ``_shape`` (property)
     - NumPy, JAX
     - Arrays of derived types
     - Outer dimensions, computed from the leaves. ``()`` for a scalar
       struct. Not stored, so it stays correct after indexing and survives
       ``jax.tree_util.tree_map``.
   * - ``__getitem__(idx)``
     - NumPy, JAX
     - Arrays of derived types
     - Indexes every leaf (recursively for nested types) and returns a new
       instance. See the note on views below.
   * - ``__setitem__(idx, value)``
     - NumPy only
     - Arrays of derived types
     - In-place assignment of every field. Fails on JAX arrays.
   * - ``with_field(name, value)``
     - JAX only
     - All derived types
     - Returns a copy with one field replaced. Field writes in the generated
       JAX code use it.
   * - ``set(idx, value)``
     - JAX only
     - Arrays of derived types
     - Functional version of ``__setitem__``, using ``.at[idx].set``.

.. note::

   ``__getitem__`` returns views for basic indices (integers, slices) and
   copies for fancy indices (lists, integer arrays, boolean masks). Chained
   writes such as ``m[[0, 2]].items.a[0] = 1.0`` therefore modify a temporary
   and are silently lost. The transpiler never emits them: reads and
   by-value passing may use ``x[i]``, while **every write is emitted as one
   flat subscript on the leaf**, e.g. ``x.a[i, j]``.

Simple Derived Types
--------------------

A derived type that is not used as an array is still a ``FortranDerivedType``
subclass, but it is instantiated directly with no outer shape (``_shape == ()``):

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

   p = point_type()

Writing a **scalar field** rebinds the struct with ``with_field``:

.. code-block:: python

   # NumPy
   p.x = 1.0
   # JAX
   p = p.with_field('x', 1.0)

Writing an **element of an array field** combines ``with_field`` and ``.at``:

.. code-block:: python

   # NumPy
   p.y[1] = 2.0
   # JAX
   p = p.with_field('y', p.y.at[1].set(2.0))

Having an array-valued field does **not** make the type an array of derived
types. The batch-axis rules below do not apply.

Arrays of Derived Types
-----------------------

An array of derived types is one object whose leaves are arrays. Outer
dimensions come first, followed by the field's own dimensions.

.. code-block:: fortran

   type laieff_type
     real :: a
     real :: b(10)
   end type

   type(laieff_type) :: laieff_fit(kjpindex, nvm)

.. code-block:: python

   laieff_fit = laieff_type.zeros((kjpindex, nvm))
   # laieff_fit.a.shape == (kjpindex, nvm)
   # laieff_fit.b.shape == (kjpindex, nvm, 10)
   # laieff_fit._shape  == (kjpindex, nvm)

Element access uses the flat form with outer indices first:

.. code-block:: python

   laieff_fit(i, j)%a       ->  laieff_fit.a[i, j]
   laieff_fit(i, j)%b(k)    ->  laieff_fit.b[i, j, k]

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
     - Property, derived from the leaves
     - Same property, preserved by ``tree_map``
   * - Element update
     - ``__setitem__``
     - ``x.set(i, value)``

Nested Derived Types
--------------------

A component may itself be a derived type. Fortran requires the inner type to
be defined first, so the generated classes are emitted in the same order.

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

Rules:

* **Scalar nested component**: ``field(default_factory=Inner)``. A bare
  ``Inner()`` default is rejected by dataclasses as a mutable default, and
  ``default_factory`` also guarantees each instance gets its own object.
* **Array of nested components**: annotated with the nested type (never
  ``list[...]``) and built with ``Inner.zeros(shape)``. This keeps the SoA
  layout at every level.
* ``Model.zeros((n,))`` expands recursively: ``items.a`` has shape
  ``(n, 5, 3)`` and ``one.a`` has shape ``(n, 3)``. ``Model.zeros((n,))._shape``
  is ``(n,)``.

Access chains are flattened. Indices from every struct level are collected,
outer first, and applied once on the leaf:

.. code-block:: python

   laieff_fit%comp%a                   ->  laieff_fit.comp.a
   laieff_fit%comp%temp(2)             ->  laieff_fit.comp.temp[2]
   model%items(i)%a(k)                 ->  model.items.a[i, k]
   model(i)%items(j)%a(k)              ->  model.items.a[i, j, k]
   laieff_fit(i,j,k)%a(i)              ->  laieff_fit.a[i, j, k, i]

Writes use the same flat form, which always updates the original array:

.. code-block:: python

   # NumPy
   model.items.a[1, 2] = 7.0
   # JAX (nested functional update; verify against the converter output)
   model = model.with_field('items',
       model.items.with_field('a', model.items.a.at[1, 2].set(7.0)))

Index adjustment (Fortran 1-based to Python 0-based) treats ``x.items.a[...]``
as one array whose rank is the outer dimensions plus every array level along
the chain. Loop variables that are already 0-based are left alone.

Reading Derived Types from Dumps
--------------------------------

Dumps write each derived-type variable as **one unformatted record**: a
scalar struct is one record, and an array of structs is one record holding
all elements. ``read_fortran_struct(ffile, var)`` reads it through a recursive
structured dtype and handles every case with one function:

* Sub-array dimensions are declared reversed and transposed back, so each
  group of dimensions (outer, nested array, leaf) keeps Fortran order.
* ``LOGICAL`` is read as 4-byte ``int32`` and converted with ``!= 0``.
* The dtype is built with ``align=True`` to match natural compiler padding.
  A wrong layout raises a record-size error instead of silently scrambling
  data.

Under JAX, the data is read into NumPy first and then converted with
``jax.tree_util.tree_map(jnp.asarray, ...)``.

JAX Registration
----------------

Every derived-type dataclass is registered with
``jax.tree_util.register_dataclass``, **inner types before outer types**.
They then work as pytrees under ``jit``, ``scan``, ``cond``, ``vmap`` and
differentiation, whether they are simple, arrays or nested. Because ``_shape``
is a property and not a field, registration ignores it and it is recomputed
after every ``tree_map``.

Translation Rules (JAX)
-----------------------

Field writes
~~~~~~~~~~~~

.. code-block:: python

   # NumPy
   laieff_fit.a[1, 0, 1] = 42.0
   # JAX
   laieff_fit = laieff_fit.with_field('a', laieff_fit.a.at[1, 0, 1].set(42.0))

Accumulations are recognized and use ``.at[...].add`` or
``.at[...].multiply``. Slices are preserved. The original object is never
modified.

.. warning::

   ``x.a.at[...].set(...)`` without rebinding the result is a no-op. Always
   use the generated ``with_field`` form.

The rules below concern arrays of derived types, since they depend on outer
(batch) dimensions.

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
delta variable so that two fields under the same mask do not collide.

Structs on ``self``
~~~~~~~~~~~~~~~~~~~

For ``self.soil_fit.a[0] = 1.0``, the first write reads ``self.soil_fit`` and
later reads and writes use the local ``soil_fit``. The struct is registered
as a modified attribute and the updated module is returned with
``eqx.tree_at``. Derived-type attributes of the global module appear in the
class metadata as ``{"kind": "dataclass", "type": ..., "is_array": ...}``.

Structs as arguments
~~~~~~~~~~~~~~~~~~~~

A struct passed to a function and written inside it is registered as a
modified argument.

Dimension inference
~~~~~~~~~~~~~~~~~~~

Declared dimensions of a field are the struct's outer dimensions plus the
field's inner dimensions, and for nested types, plus the dimensions of each
array level along the chain. Indexing with a scalar removes the corresponding
active dimension.

Numerical Behavior
------------------

The test suite checks that:

* ``float64`` fields stay ``float64`` in ``scan`` carries.
* Updated structs can be returned from ``jit`` and carried through ``scan``
  and ``cond`` (both branches must have the same structure).
* ``jax.grad``, ``jax.jvp`` and ``jax.vjp`` work through field writes and
  through ``x[ji, ...]`` under ``vmap``.
* ``_shape`` survives ``tree_map``, ``__getitem__`` and nesting.

Limitations
-----------

* **Deferred-shape components** (``ALLOCATABLE`` or ``(:)`` inside a derived
  type) are not supported and raise ``NotImplementedError``. Every component
  shape must be known when the class is defined.
* Pointer components and polymorphic types (``CLASS``) are not supported.
* Initializers on arrays of derived types (``[t(1,2), t(3,4)]``) are not
  supported.
* ``x[i] = value`` and chained writes fail or are lost on the JAX object. The
  converter emits ``with_field`` or ``set`` instead.
* Dumps must store each variable as one record in the layout described above.
