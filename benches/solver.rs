use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ndarray::*;
use ndarray_ode::prelude::*;

pub fn benchmarking(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solver");

    for h in [0.01, 0.001] {
        #[allow(non_snake_case)]
        let T = 10.0;
        let x0 = array![10.0, 1.0];

        group.bench_function(BenchmarkId::new("explicit euler", h), |b| {
            b.iter(|| explicit_euler(x0.view(), h, T))
        });
        group.bench_function(BenchmarkId::new("implicit euler", h), |b| {
            b.iter(|| implicit_euler(x0.view(), h, T))
        });
        group.bench_function(BenchmarkId::new("symplectic euler", h), |b| {
            b.iter(|| symplectic_euler(x0.view(), h, T))
        });
    }
    group.finish();
}

criterion_group!(benches, benchmarking);
criterion_main!(benches);

#[allow(non_snake_case)]
fn explicit_euler(x0: ArrayView1<f64>, h: f64, T: f64) {
    let ex_euler = ExplicitEuler::new(h, undamped_oscilator_f64);
    let mut ode = Ode::explicit(ex_euler, x0.to_owned());
    ode.set_step_size(h).set_t(T).set_with_tqdm(false);
    ode.run();
}

#[allow(non_snake_case)]
fn implicit_euler(x0: ArrayView1<f64>, h: f64, T: f64) {
    let im_euler = ImplicitEuler::new(h, undamped_oscilator_ad);
    let mut ode = Ode::implicit(im_euler, x0.to_owned());
    ode.set_step_size(h).set_t(T).set_with_tqdm(false);
    ode.run();
}

#[allow(non_snake_case)]
fn symplectic_euler(x0: ArrayView1<f64>, h: f64, T: f64) {
    let sym_euler = SymplecticEuler::new(h, undamped_oscilator_ad);
    let mut ode = Ode::implicit(sym_euler, x0.to_owned());
    ode.set_step_size(h).set_t(T).set_with_tqdm(false);
    ode.run();
}

fn undamped_oscilator_ad(x: ArrayView1<AD>) -> Array1<AD> {
    let m = 1.0;
    let c = 1.0;
    let (q, p) = (x[0], x[1]);
    array![p / m, -q / c]
}

fn undamped_oscilator_f64(x: ArrayView1<f64>) -> Array1<f64> {
    let m = 1.0;
    let c = 1.0;
    let (q, p) = (x[0], x[1]);
    array![p / m, -q / c]
}
