mod config_write;

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use ndarray::{Array, Array2, Array3, Axis, ShapeBuilder};
use config_write::{build_config, get};

#[derive(Debug, Copy, Clone)]
pub(crate) enum Dimension {
    X,
    #[allow(dead_code)]
    Y,
    #[allow(dead_code)]
    Z,
}

struct FileInfo {
    nx: usize,
    ny: usize,
    nz: usize,
    file_prefix: String,
}

impl FileInfo {
    pub fn new(config: &HashMap<String, Option<String>>, dimension: Dimension, file_prefix: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self{
            nx: get(config,"Nx").parse()?,
            ny: get(config,"Ny").parse()?,
            nz: get(config,"Nz").parse()?,
            file_prefix: format!("{file_prefix}_{dimension}_mean"),
        })
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    pub fn prefix(&self) -> &str {
        &self.file_prefix
    }
}

struct MeanWriter {
    writer: BufWriter<File>,
}

impl MeanWriter {
    pub fn new(file_prefix: &str,index: usize) -> Self {
        let file_name = format!("{file_prefix}.{index}");
        let writer = BufWriter::new(File::create(&file_name).unwrap());
        Self {
            writer,
        }
    }

    pub fn write_mean(&mut self,output: Array2<f64>) -> Result<(), Box<dyn Error>> {
        for &v in output.iter() {
            self.writer.write_all(&v.to_le_bytes())?;        // little-endian f64
        }
        Ok(self.writer.flush()?)
    }
}
pub(crate) fn compute_and_save_mean(
    files: Vec<PathBuf>,
    config: &HashMap<String, Option<String>>,
    dimension: Dimension,
) -> Result<(), Box<dyn Error>> {

    let file_info = FileInfo::new(config, dimension, "t")?;

    files.iter().enumerate()
        .map(|(i, f)| {

            let array = read_values_from_file(&file_info, f)?;

            let mean_array = do_mean(array, dimension);

            let mut writer = MeanWriter::new(file_info.prefix(), i);

            writer.write_mean(mean_array)?;

            Ok(())
        })
        .collect::<Result<(), Box<dyn Error>>>()?;

    build_config(config,dimension,&file_info.file_prefix)?;
    Ok(())
}

fn do_mean(array: Array3<f64>, mean_dimension: Dimension) -> Array2<f64> {

    let mean_dimension = match mean_dimension {
        Dimension::X => 2,
        Dimension::Y => 0,
        Dimension::Z => 1,
    };

    let mean_array = match array.mean_axis(Axis(mean_dimension)) {
        Some(array) => array,
        None => panic!("Something went wrong while taking the average...\
                This is a bug, please report"),
    };
    mean_array
}

fn read_values_from_file(file_info: &FileInfo, f: &PathBuf) -> Result<Array3<f64>, Box<dyn Error>> {
    let t0 = std::fs::read(&f)?;
    let floats: &[f64] = bytemuck::cast_slice(&t0);
    // TODO this makes no effort to reorganize the file contents.
    // Need a function that undoes the efforts of tests::to_spins_ordering
    // perhaps call it undo_spins_ordering
    let mut array = Array3::from_shape_vec(file_info.shape(), floats.to_vec())?;
    array.permute_axes([1,2,0]);
    Ok(array)
}

impl FromStr for Dimension {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "x" | "X" => Ok(Dimension::X),
            "y" | "Y" => todo!("Only operations along X are implemented"),
            "z" | "Z" => todo!("Only operations along X are implemented"),
            _ => Err(format!("Unknown dimension: {}", s)),
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dimension::X => "X",
            Dimension::Y => "Y",
            Dimension::Z => "Z",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::io::Write;
    use std::path::PathBuf;
    use ndarray::{Array3, ShapeBuilder};
    use tempfile;
    use tempfile::TempDir;
    use crate::mean::{compute_and_save_mean, Dimension, read_values_from_file, FileInfo, do_mean};

    fn create_fake_config() -> HashMap<String, Option<String>> {
        let mut config = HashMap::new();
        config.insert("Lx".to_string(),Some("1.0".to_string()));
        config.insert("Ly".to_string(),Some("2.0".to_string()));
        config.insert("Lz".to_string(),Some("3.0".to_string()));
        config.insert("Nx".to_string(),Some("3".to_string()));
        config.insert("Ny".to_string(),Some("4".to_string()));
        config.insert("Nz".to_string(),Some("5".to_string()));
        config
    }

    fn write<F: FnOnce() -> Vec<f64>>(path: &PathBuf, f: F) -> Result<(), Box<dyn Error>> {
        let mut file = std::fs::File::create(&path)?;
        let values = f();
        for v in &values {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
    #[test]
    fn read_to_array() {
        let (config, path, _dir) = create_fake_file();

        let file_info = FileInfo::new(&config,Dimension::X,"b").unwrap();
        let array = read_values_from_file(&file_info, &path).unwrap();
        let array = array.as_standard_layout();
        insta::assert_yaml_snapshot!(array.as_slice());
    }

    #[test]
    fn compute_mean() {
        let i = 0usize;
        let nx = 3usize;
        let ny = 4usize;
        let nz = 5usize;
        let length = nx * ny * nz;
        let min = i * length;
        let max = (i + 1) * length;
        let values: Vec<f64> = (min..max).map(|v| v as f64).collect();
        let array = Array3::from_shape_vec((ny,nz,nx).f(), values).unwrap();
        let mean_dimension = Dimension::Z;

        let array = do_mean(array, mean_dimension);
        let as_vec = array.as_slice();

        assert_eq!(array.len(), nx * ny);
        //assert_eq!(as_vec[])
    }

    /// This tests the
    #[test]
    fn spins_ordering_test() {
        let nx = 2usize;
        let ny = 3usize;
        let nz = 5usize;
        let write_foo = || {
            let length = nx * ny * nz;

            let arr = Array3::<f64>::from_shape_fn(
                (nz, ny, nx),
                |(z, y, x)| {
                    (x + nx*y + nx*ny*z) as f64
                },
            );
            let flat = arr.as_slice().unwrap();

            let values = to_spins_ordering((nx, ny, nz), flat);
            values
        };
        let values = write_foo();
        insta::assert_yaml_snapshot!(values);
    }

    fn to_spins_ordering((nx,ny,nz): (usize,usize,usize), flat: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0f64; nx * ny * nz];

        for x in 0..nx {
            for z in 0..nz {
                for y in 0..ny {
                    let xyz_idx =
                        x +
                            nx * y +
                            nx * ny * z;

                    let file_idx =
                        y +
                            ny * z +
                            ny * nz * x;

                    values[file_idx] = flat[xyz_idx];
                }
            }
        }
        values
    }

    fn create_fake_file() -> (HashMap<String, Option<String>>, PathBuf, TempDir) {
        let config = create_fake_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.0");
        let nx = 3usize;
        let ny = 4usize;
        let nz = 5usize;
        let write_foo = || {
            let length = nx * ny * nz;
            let values = (0..length).map(|v| v as f64).collect::<Vec<_>>();
            let values = to_spins_ordering((nx,ny,nz),&values);
            values
        };

        write(&path, write_foo).unwrap();
        (config, path, dir)
    }
}
