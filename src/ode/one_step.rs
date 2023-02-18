use crate::{ad::*, ode::*};
use indicatif::ProgressIterator;
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
    with_progress: bool,
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
    #[allow(non_snake_case)]
    fn execute(
        &mut self,
        t: usize,
        x0: Array1<AD>,
        result: &mut Array2<f64>,
        time: &mut Vec<f64>,
        J: &mut Array2<f64>,
        slope_buffer: &mut Array1<AD>,
    ) -> Array1<AD> {
        self.scheme.update(x0.clone());
        match newton(f64::EPSILON, &self.scheme, x0.clone(), J, slope_buffer) {
            Ok(x1) => {
                result
                    .row_mut(t)
                    .iter_mut()
                    .zip(x1.iter())
                    .for_each(|(x, &y)| *x = y.x());
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

    #[allow(non_snake_case)]
    fn run(mut self) -> (Vec<f64>, Array2<f64>) {
        let n: f64 = self.T / self.h;
        let n = n.floor() as usize;
        let mut x0 = self.initial.clone();

        let mut result: Array2<f64> = Array::zeros((n, x0.len()));
        result
            .row_mut(0)
            .iter_mut()
            .zip(self.initial.to_f64().iter())
            .for_each(|(x, &y)| *x = y); //(self.initial.to_f64().view()).unwrap();
        let mut time = vec![0.0; n];

        let l = x0.len();
        let mut J = Array2::zeros((l, l));
        let mut slope_buffer = x0.clone();

        if self.with_progress {
            for t in tqdm(1..n) {
                x0 = self.execute(t, x0, &mut result, &mut time, &mut J, &mut slope_buffer);
            }
        } else {
            for t in 1..n {
                x0 = self.execute(t, x0, &mut result, &mut time, &mut J, &mut slope_buffer);
            }
        }

        (time, result)
    }

    fn set_with_progress(&mut self, with_progress: bool) -> &mut Self {
        self.with_progress = with_progress;
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
            with_progress: true,
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
    with_progress: bool,
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

        let mut result: Array2<f64> = Array::zeros((n, x0.len()));
        result
            .row_mut(0)
            .iter_mut()
            .zip(x0.iter())
            .for_each(|(x, &y)| *x = y);
        let mut time = vec![0.0; n];

        if self.with_progress {
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

    fn set_with_progress(&mut self, with_progress: bool) -> &mut Self {
        self.with_progress = with_progress;
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
        // let mut x1 = result.row_mut(t);

        // self.scheme.next1(x0.view(), x1);

        let x1 = self.scheme.next(x0.view());
        result
            .row_mut(t)
            .iter_mut()
            .zip(x1.iter())
            .for_each(|(x, &y)| *x = y);
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
            with_progress: true,
        }
    }
}
