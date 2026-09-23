# Multi-Module Fortran Demo

A small Fortran project that demonstrates a realistic module dependency graph, pure functions, text and binary I/O, and an implicit tridiagonal solver, wrapped around a 1-D energy balance climate model.

## Layout

```
.
├── Makefile
├── mod_parameters.f90
├── mod_grid.f90
├── mod_math.f90
├── mod_io.f90
├── mod_simulation.f90
├── main_program.f90
└── README.md
```

Build outputs: `obj/`, `mod/`, `ebm`, `profile.txt`, `state.bin`.

## Dependency graph

```
mod_parameters
   ├── mod_grid
   ├── mod_math
   └── mod_io
          │
          ▼
     mod_simulation
          │
          ▼
      main_program
```

No cycles. `mod_parameters` sits at the bottom; `mod_simulation` uses all three siblings; `main_program` is a thin driver.

## Physics

For each latitude band `i`:

```
C dT/dt = S(i) (1 - α(i)) - (A + B T(i)) + d/dy ( D dT/dy )
```

- `C` heat capacity, `S` insolation, `α` albedo (ice-albedo feedback), `A + B T` outgoing longwave, `D` meridional diffusivity.
- Diffusion is integrated implicitly and solved with the Thomas algorithm — unconditionally stable, 30-day time step.

## Modules

| Module | Provides |
|---|---|
| `mod_parameters` | kinds, physical constants, grid size |
| `mod_grid` | latitude, area weights, insolation, `surface_albedo` |
| `mod_math` | `mean`, `variance`, `stddev`, `weighted_mean`, `median`, `correlation`, `clamp`, `linspace`, `solve_tridiag` (all pure) |
| `mod_io` | text and binary read/write, profile table, state snapshot |
| `mod_simulation` | `initialize_state`, `step`, `diagnose`, `energy_budget`, `report`, `verify` |
| `main_program` | driver |

## Build and run

```bash
make            # build ./ebm
make run        # build and execute
make clean      # remove artifacts
make restore    # restore .f90 from _org.fgpt backups
```

## Output

| File | Content |
|---|---|
| `profile.txt` | one row per latitude band: `lat`, `T`, `OLR`, `transport` |
| `state.bin` | binary snapshot: `n`, `temperature`, `F_olr`, `F_transport` |
