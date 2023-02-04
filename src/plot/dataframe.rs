use std::fs::File;

use super::Series;

pub struct Dataframe<'a> {
    data: Vec<Series<'a>>,
}
impl<'a> Dataframe<'a> {
    pub fn new() -> Self {
        Dataframe { data: vec![] }
    }
    pub fn push(&mut self, series: Series<'a>) {
        self.data.push(series);
    }
    pub fn store(&self, file: &mut File) -> Result<(), polars::prelude::PolarsError> {
        use polars::prelude::*;

        let columns = self
            .data
            .iter()
            .map(|x| Series::new(x.name, x.values.clone()))
            .collect();
        let mut df = DataFrame::new(columns)?;
        ParquetWriter::new(file).finish(&mut df)?;
        Ok(())
    }
}

impl<'a> Default for Dataframe<'a> {
    fn default() -> Self {
        Self::new()
    }
}
