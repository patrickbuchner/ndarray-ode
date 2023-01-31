use std::path::Path;

use ndarray::*;
use polars::prelude::*;
use ndarray_ode::{ad::*, plot, ode::{*, solver::SymplecticEuler}};

const DOF: usize = 4;
#[allow(non_upper_case_globals)]
const μ: f64 = 1.;
fn main() -> PolarsResult<()> {
    let h = 0.05;
    #[allow(non_snake_case)]
    let T = 20.0;
    let x0 = array![1.0, 0.0, 0.0, 1.0];
    assert![x0.len() == DOF];

    let euler = SymplecticEuler::new(x0.to_ad(), h, keppler);
    let mut ode = Ode::new(euler, x0.to_ad());
    ode.set_step_size(h).set_t(T);

    let (time, result) = ode.run();

    let current_dir = Path::new(".");
    let folder = current_dir.join("examples").join("keppler");
    let file = std::fs::File::create(folder.join("keppler.parquet"))?;
    store(time, result, file)?;
    plot::python(folder.join("plot_keppler.py"), "");
    Ok(())
}

fn store(
    time: Vec<f64>,
    result: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mut file: std::fs::File,
) -> Result<(), PolarsError> {
    let t = Series::new("t", time);
    let x = Series::new("x", result.column(0).to_vec());
    let y = Series::new("y", result.column(1).to_vec());
    let px = Series::new("px", result.column(2).to_vec());
    let py = Series::new("py", result.column(3).to_vec());
    let mut df = DataFrame::new(vec![t, x, y, px, py])?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    Ok(())
}
fn keppler(x: ArrayView1<AD>) -> Array1<AD> {
    let (x, y, px, py) = (x[0], x[1], x[2], x[3]);
    let factor = -μ * (x * x + y * y).powi(3).sqrt();
    array![px, py, factor * x, factor * y]
}
