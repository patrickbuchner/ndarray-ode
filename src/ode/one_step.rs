use crate::{ad::*, ode::*};
use ndarray::*;
use tqdm::tqdm;

/// This is the most classical ode solver.
/// From the last known step it extrapolates with a residual function, which can be for example any Runge-Kutta-scheme, to the next time step.
/// Currently it only supports fixed timesteps `h`.

#[allow(non_snake_case)]
pub struct OdeIm<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual1Step,
{
    scheme: Scheme,
    initial: Array1<AD>,
    h: f64,
    T: f64,
    ɛ: f64,
    with_tqdm: bool,
}
impl<Scheme> OdeIm<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual1Step,
{
    pub fn set_epsilon(&mut self, ɛ: f64) -> &mut Self {
        self.ɛ = ɛ;
        self
    }
    #[inline]
    fn execute(
        &mut self,
        t: usize,
        x0: Array1<f64>,
        result: &mut Array2<f64>,
        time: &mut Vec<f64>,
    ) -> Array1<f64> {
        self.scheme.update(x0.to_ad());
        match newton(f64::EPSILON, &self.scheme, x0.clone()) {
            Ok(x1) => {
                result.push_row(x1.view()).unwrap();
                time[t] = t as f64 * self.h;
                x1
            }
            Err(_) => todo!(),
        }
    }
}

impl<Scheme> ODE<Scheme> for OdeIm<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual1Step,
{
    fn set_step_size(&mut self, h: f64) -> &mut Self {
        self.h = h;
        self
    }

    #[allow(non_snake_case)]
    fn set_t(&mut self, T: f64) -> &mut Self {
        self.T = T;
        self
    }

    fn run(mut self) -> (Vec<f64>, Array2<f64>) {
        let n: f64 = self.T / self.h;
        let n = n.floor() as usize;
        let mut x0 = self.initial.clone().to_f64();

        let mut result: Array2<f64> = Array::zeros((0, x0.len()));
        result.push_row(self.initial.to_f64().view()).unwrap();
        let mut time = vec![0.0; n];

        if self.with_tqdm {
            for t in tqdm(1..n) {
                x0 = self.execute(t, x0, &mut result, &mut time);
            }
        } else {
            for t in 1..n {
                x0 = self.execute(t, x0, &mut result, &mut time);
            }
        }

        (time, result)
    }

    fn set_with_tqdm(&mut self, with_tqdm: bool) -> &mut Self {
        self.with_tqdm = with_tqdm;
        self
    }
}
impl<Scheme> OneStep for OdeIm<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual1Step,
{
    type Scheme = Scheme;
    /// Create the ode solver with a start value and a residual function to minimize.
    fn new(scheme: Self::Scheme, initial: Array1<f64>) -> Self {
        OdeIm {
            scheme,
            initial: initial.to_ad(),
            h: 0.1,
            T: 1.0,
            ɛ: f64::EPSILON,
            with_tqdm: true,
        }
    }
}
#[allow(non_snake_case)]
pub struct OdeEx<Scheme>
where
    Scheme: Explicit + std::marker::Sync,
{
    scheme: Scheme,
    initial: Array1<f64>,
    h: f64,
    T: f64,
    with_tqdm: bool,
}

impl<Scheme> ODE<Scheme> for OdeEx<Scheme>
where
    Scheme: Explicit + std::marker::Sync,
{
    fn set_step_size(&mut self, h: f64) -> &mut Self {
        self.h = h;
        self
    }

    #[allow(non_snake_case)]
    fn set_t(&mut self, T: f64) -> &mut Self {
        self.T = T;
        self
    }

    fn run(self) -> (Vec<f64>, Array2<f64>) {
        let n: f64 = self.T / self.h;
        let n = n.floor() as usize;
        let mut x0 = self.initial.clone();

        let mut result: Array2<f64> = Array::zeros((0, x0.len()));
        result.push_row(self.initial.view()).unwrap();
        let mut time = vec![0.0; n];

        if self.with_tqdm {
            for t in tqdm(1..n) {
                x0 = self.execute(t, x0, &mut result, &mut time);
            }
        } else {
            for t in 1..n {
                x0 = self.execute(t, x0, &mut result, &mut time);
            }
        }
        (time, result)
    }

    fn set_with_tqdm(&mut self, with_tqdm: bool) -> &mut Self {
        self.with_tqdm = with_tqdm;
        self
    }
}

impl<Scheme> OdeEx<Scheme>
where
    Scheme: Explicit + std::marker::Sync,
{
    #[inline]
    fn execute(
        &self,
        t: usize,
        x0: Array1<f64>,
        result: &mut Array2<f64>,
        time: &mut Vec<f64>,
    ) -> Array1<f64> {
        let x1 = self.scheme.next(x0.view());
        result.push_row(x1.view()).unwrap();
        time[t] = t as f64 * self.h;
        x1
    }
}

impl<Scheme> OneStep for OdeEx<Scheme>
where
    Scheme: Explicit + std::marker::Sync,
{
    type Scheme = Scheme;
    fn new(scheme: Self::Scheme, initial: Array1<f64>) -> Self {
        OdeEx {
            scheme,
            initial,
            h: 0.1,
            T: 1.0,
            with_tqdm: true,
        }
    }
}
