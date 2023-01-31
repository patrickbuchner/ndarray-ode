pub struct Series<'a> {
    pub name: &'a str,
    pub values: Vec<f64>,
}

impl<'a> Series<'a> {
    pub fn new(name: &'a str, val: &[f64]) -> Self {
        let values: Vec<f64> = val.to_vec();
        Series { name, values }
    }
}
