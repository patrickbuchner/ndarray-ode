# ndarray-ode
Ordinaray differential equation solver for ndarray.

## Implemented
* One Step ODE solver (can use Runge Kutta methods)
* Two Step Methods (Midpoint rules)

## Implementation inspirations from
* [peroxide](https://crates.io/crates/peroxide) version 0.32.1 (Automatic differentiation and ode)
* [argmin](https://crates.io/crates/argmin) version 0.8.0 (Splitting of ODE and Residual)
* Lectures at [Friedrich Alexander Universität](https://www.tf.fau.de/)

## Examples
See the [examples](https://github.com/patrickbuchner/ndarray-ode/tree/main/examples) directory.
One can run for example: 
```
cargo r -p keppler
```
## Details
Plotting works via python matplotlib and pyarrow.
Storing of calculated data can be done with [polars](https://github.com/pola-rs/polars).

## To be done
* Documentation
* Further classic integrators (currently only symplectic Euler)
