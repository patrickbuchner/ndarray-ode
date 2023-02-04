use std::path::Path;

use ndarray::*;
use ndarray_ode::prelude::*;
use rayon::prelude::*;

const DOF: usize = 4;
#[allow(non_upper_case_globals)]
const μ: f64 = 1.;
fn main() {
    let h = 0.0005;
    #[allow(non_snake_case)]
    let T = 2000.0;
    let x0 = array![1.0, 0.0, 0.0, 1.0];
    assert![x0.len() == DOF];

    let ode = vec![OdeType::SymplecticEuler, OdeType::ImplicitEuler, OdeType::Expliciteuler];

    let current_dir = Path::new(".");
    let folder = current_dir.join("examples").join("keppler");
    run(ode, x0, h, T, &folder);

    plot::python(folder.join("plot_keppler.py"), &vec![""]);
}
#[allow(nonstandard_style)]
fn run(
    ode: Vec<OdeType>,
    x0: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    h: f64,
    T: f64,
    folder: &std::path::PathBuf,
) {
    ode.par_iter().for_each(|e| match e {
        OdeType::SymplecticEuler => {
            let euler = SymplecticEuler::new(x0.to_ad(), h, keppler);
            let mut ode = Ode::new(euler, x0.to_ad());
            ode.set_step_size(h).set_t(T);

            let (time, result) = ode.run();
            let file = std::fs::File::create(folder.join("keppler_symplectic.parquet")).unwrap();
            store(time, result, file);
        }
        OdeType::ImplicitEuler => {
            let euler = ImplicitEuler::new(x0.to_ad(), h, keppler);
            let mut ode = Ode::new(euler, x0.to_ad());
            ode.set_step_size(h).set_t(T);

            let (time, result) = ode.run();
            let file = std::fs::File::create(folder.join("keppler_implicit.parquet")).unwrap();
            store(time, result, file);
        }
        OdeType::Expliciteuler => {
            let euler = ExplicitEuler::new(x0.to_ad(), h, keppler);
            let mut ode = Ode::new(euler, x0.to_ad());
            ode.set_step_size(h).set_t(T);

            let (time, result) = ode.run();
            let file = std::fs::File::create(folder.join("keppler_explicit.parquet")).unwrap();
            store(time, result, file);
        }
    });
}

enum OdeType {
    SymplecticEuler,
    ImplicitEuler,
    Expliciteuler,
}

fn store(
    time: Vec<f64>,
    result: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mut file: std::fs::File,
) {
    let mut df = plot::Dataframe::new();
    df.push(plot::Series::new("t", &time));
    df.push(plot::Series::new("x", &result.column(0).to_vec()));
    df.push(plot::Series::new("y", &result.column(1).to_vec()));
    df.push(plot::Series::new("px", &result.column(2).to_vec()));
    df.push(plot::Series::new("py", &result.column(3).to_vec()));
    df.store(&mut file).unwrap();
}
fn keppler(x: ArrayView1<AD>) -> Array1<AD> {
    let (x, y, px, py) = (x[0], x[1], x[2], x[3]);
    let factor = -μ * (x * x + y * y).powi(3).sqrt();
    array![px, py, factor * x, factor * y]
}
