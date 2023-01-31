pub fn open_png(path_to_plot: &str) {
    use std::process::Command;

    #[cfg(target_family = "unix")]
    Command::new("/usr/bin/xdg-open")
        .arg(path_to_plot)
        .output()
        .unwrap();
}

pub fn python(file: std::path::PathBuf, method_name: &str) {
    use std::io::Write;
    use std::process::Command;
    let output = Command::new("python").arg(file).arg(method_name).output();
    match output {
        Ok(o) => {
            if !o.status.success() {
                std::io::stdout().write_all(&o.stdout).unwrap();
                std::io::stderr().write_all(&o.stderr).unwrap();
            }
        }
        Err(e) => {
            println!("{e:#}");
        }
    }
}

mod dataframe;
pub use dataframe::Dataframe;
mod series;
pub use series::Series;