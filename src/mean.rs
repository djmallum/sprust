mod config_write;

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use ndarray::{Array2, Array3, Axis, ShapeBuilder};
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
        dbg!(&file_name);
        let writer = BufWriter::new(File::create(&file_name).unwrap());
        dbg!(&writer);
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
    file_prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let first_file = files.first().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cannot compute means without input files",
        )
    })?;

    let output_dir = first_file.parent().unwrap_or_else(|| Path::new("."));
    let file_info = FileInfo::new(config, dimension, file_prefix)?;
    let output_prefix = output_dir.join(file_info.prefix());
    let output_prefix = output_prefix.to_string_lossy();

    files.iter().enumerate()
        .map(|(i, file)| {
            let array = read_values_from_file(&file_info, file)?;
            let mean_array = do_mean(array, dimension);

            let mut writer = MeanWriter::new(output_prefix.as_ref(), i);
            writer.write_mean(mean_array)?;

            Ok(())
        })
        .collect::<Result<(), Box<dyn Error>>>()?;

    build_config(config, dimension, output_prefix.as_ref())?;
    //let file_info = FileInfo::new(config, dimension, file_prefix)?;

    //files.iter().enumerate()
    //    .map(|(i, f)| {

    //        let array = read_values_from_file(&file_info, f)?;

    //        let mean_array = do_mean(array, dimension);

    //        let mut writer = MeanWriter::new(file_info.prefix(), i);
    //        writer.write_mean(mean_array)?;

    //        Ok(())
    //    })
    //    .collect::<Result<(), Box<dyn Error>>>()?;

    //build_config(config,dimension,&file_info.file_prefix)?;
    Ok(())
}

fn do_mean(array: Array3<f64>, mean_dimension: Dimension) -> Array2<f64> {

    let mean_dimension = match mean_dimension {
        Dimension::X => 0,
        Dimension::Y => 1,
        Dimension::Z => 2,
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
    let (nx, ny, nz) = file_info.shape();
    let mut array = Array3::from_shape_vec((ny,nz,nx).f(), floats.to_vec())?;
    array.permute_axes([2,0,1]);
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
    use std::path::{PathBuf};
    use ndarray::{Array2, Array3};
    use tempfile;
    use crate::mean::{Dimension, read_values_from_file, FileInfo, MeanWriter, compute_and_save_mean};

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
    fn spins_ordering_test() {
        let nx = 2usize;
        let ny = 3usize;
        let nz = 5usize;
        let write_foo = || {

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

    mod mean_calculation {
        use ndarray::{Array2, Array3, Ix2};
        use crate::mean::{do_mean, Dimension};

        /// Test of a calculation of the mean along the x-axis.
        ///
        /// Input array is a*x + b*y + c*z.
        ///
        /// The mean along x is
        ///
        /// a/nx * \sum_{i=0}^N(x_i) + b*y + c*z
        #[test]
        fn x() {
            let (nx, ny, nz) = (3, 5, 7);
            let (a, b, c) = (1.0, 2.0, 3.0);
            let input = Array3::from_shape_fn(
                (nx, ny, nz),
                |(x, y, z)| {
                    a * x as f64 + b * y as f64 + c * z as f64
                }
            );
            let expected = ExpectedBuilder{
                mean_coeff: a,
                other_coeff: (b, c),
                mean_dim_length: nx,
                other_lengths: (ny, nz),
            }.get();

            let dimension = Dimension::X;

            let result = do_mean(input,dimension);

            assert_eq!(expected, result);
        }

        /// See doc of x(), same logic
        #[test]
        fn y() {
            let (nx, ny, nz) = (3, 5, 7);
            let (a, b, c) = (1.0, 2.0, 3.0);
            let input = Array3::from_shape_fn(
                (nx, ny, nz),
                |(x, y, z)| {
                    a * x as f64 + b * y as f64 + c * z as f64
                }
            );

            let expected = ExpectedBuilder{
                mean_coeff: b,
                other_coeff: (a, c),
                mean_dim_length: ny,
                other_lengths: (nx, nz),
            }.get();

            let dimension = Dimension::Y;

            let result = do_mean(input,dimension);

            assert_eq!(expected, result);
        }

        /// See doc of x(), same logic
        #[test]
        fn z() {
            let (nx, ny, nz) = (3, 5, 7);
            let (a, b, c) = (1.0, 2.0, 3.0);
            let input = Array3::from_shape_fn(
                (nx, ny, nz),
                |(x, y, z)| {
                    a * x as f64 + b * y as f64 + c * z as f64
                }
            );

            let expected = ExpectedBuilder{
                mean_coeff: c,
                other_coeff: (a, b),
                mean_dim_length: nz,
                other_lengths: (nx, ny),
            }.get();

            let dimension = Dimension::Z;

            let result = do_mean(input,dimension);

            assert_eq!(expected, result);
        }
        struct ExpectedBuilder {
            mean_coeff: f64,
            other_coeff: (f64,f64),
            mean_dim_length: usize,
            other_lengths: (usize,usize),
        }

        impl ExpectedBuilder {
            pub fn get(&self) -> Array2<f64> {
                Array2::from_shape_fn(
                    self.other_lengths,
                    |(x, y)| {
                        let avgd_values = (0..self.mean_dim_length).map(|v| {
                            let v = v as f64;
                            self.mean_coeff * v / self.mean_dim_length as f64
                        }).sum::<f64>();
                        avgd_values
                            + (self.other_coeff.0 * x as f64)
                            + (self.other_coeff.1 * y as f64)
                    }
                )
            }
        }

        /// Test of a two-dimensional input with a single dimension with length 1
        ///
        /// `expected` is built by simply squeezing the input array
        #[test]
        fn two_dimensions() {
            let (nx,ny, nz) = (3, 1, 7);
            let (a, b, c) = (1.0, 2.0, 3.0);
            let input = Array3::from_shape_fn(
                (nx, ny, nz),
                |(x, y, z)| {
                    a * x as f64 + b * y as f64 + c * z as f64
                }
            );

            let expected = input.clone()
                .into_dyn()
                .squeeze()
                .into_dimensionality::<Ix2>().unwrap();

            let dimension = Dimension::Y;

            let result = do_mean(input,dimension);

            assert_eq!(expected, result);
        }
    }

    /// Testing output
    #[test]
    fn output() {
        let (nx, nz) = (3, 7);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a_b");
        let vec_copy = (0..21).map(|i| i as f64).collect::<Vec<_>>();
        let mean_array = Array2::from_shape_vec(
            (nx, nz),
            vec_copy.clone()
        ).unwrap();
        let mut writer = MeanWriter::new(path.to_str().unwrap(), 999);
        let path = dir.path().join("a_b.999");

        writer.write_mean(mean_array).unwrap();

        assert!(path.exists());
        let t0 = std::fs::read(&path).unwrap();
        let floats: &[f64] = bytemuck::cast_slice(&t0);

        insta::assert_yaml_snapshot!(floats);
    }

    fn encoded(x: usize, y: usize, z: usize) -> f64 {
        (100 * x + 10 * y + z) as f64
    }

    // MATLAB memmapfile Format [Ny, Nz, Nx]:
    // y varies fastest, then z, then x.
    fn spins_values(nx: usize, ny: usize, nz: usize) -> Vec<f64> {
        (0..nx)
            .flat_map(|x| {
                (0..nz).flat_map(move |z| {
                    (0..ny).map(move |y| encoded(x, y, z))
                })
            })
            .collect()
    }

    #[test]
    fn read_to_array_preserves_coordinates() {
        let (nx, ny, nz) = (3, 4, 5);
        let config = create_fake_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.0");

        write(&path, || spins_values(nx, ny, nz)).unwrap();

        let file_info = FileInfo::new(&config, Dimension::X, "b").unwrap();
        let array = read_values_from_file(&file_info, &path).unwrap();

        assert_eq!(array.shape(), &[nx, ny, nz]);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    assert_eq!(
                        array[[x, y, z]],
                        encoded(x, y, z),
                        "wrong value at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }
    #[test]
    fn compute_mean_and_save() {
        let config = create_fake_config();
        let dir = tempfile::tempdir().unwrap();
        let paths = vec![dir.path().join("b.0"), dir.path().join("b.1")
            , dir.path().join("b.2"), dir.path().join("b.3")
            , dir.path().join("b.4"), dir.path().join("b.5")];

        let nx: usize = config.get("Nx").unwrap().as_ref().unwrap().parse().unwrap();
        let ny: usize = config.get("Ny").unwrap().as_ref().unwrap().parse().unwrap();
        let nz: usize = config.get("Nz").unwrap().as_ref().unwrap().parse().unwrap();

        paths.iter().for_each(|path| {
            write(&path, || spins_values(nx, ny, nz)).unwrap();
        });

        compute_and_save_mean(paths,&config,Dimension::Y,"b").unwrap();

        let expected_paths = vec![dir.path().join("b_Y_mean.0"), dir.path().join("b_Y_mean.1"),
            dir.path().join("b_Y_mean.2"), dir.path().join("b_Y_mean.3"), dir.path().join("b_Y_mean.4"),
            dir.path().join("b_Y_mean.5")];

        let expected_len = (nx * nz * size_of::<f64>()) as u64;

        expected_paths.iter().for_each(|path| {
            assert!(path.exists(), "expected output file to exist: {}", path.display());

            let actual_len = std::fs::metadata(path).unwrap().len();
            assert_eq!(
                actual_len,
                expected_len,
                "unexpected file length for {}",
                path.display()
            );
        });

    }
}
