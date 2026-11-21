use crate::mean::Dimension;
use crate::config_read::new_ini;
use std::collections::HashMap;
use std::error::Error;

pub(crate) fn build_config(
    config: &HashMap<String, Option<String>>,
    dimension: Dimension,
    file_prefix: &str,
) -> Result<(),Box<dyn Error>> {
    let mut ini = new_ini(config)?;
    let key = match dimension {
        Dimension::X => "Nx",
        Dimension::Y => "Ny",
        Dimension::Z => "Nz",
    };
    ini.set("default", key, Some("1".into()));

    ini.write(format!("{file_prefix}.conf"))?;
    Ok(())
}

pub(crate) fn get<'a>(config: &'a HashMap<String, Option<String>>, var_name: &str) -> &'a str {
    let nx = config.get(var_name).and_then(|o| o.as_ref());
    let nx = match nx {
        Some(val) => val,
        None => panic!(),
    };
    nx
}
