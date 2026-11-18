use std::collections::HashMap;
use configparser::ini::Ini;

pub(crate) fn new() -> Result<HashMap<String, HashMap<String, Option<String>>>,Box<dyn std::error::Error>> {
    let mut cfg = Ini::new_cs();            // lowercases keys
    cfg.set_comment_symbols(&['#']);

    let file_name: String = "spins.conf".to_string();
    let content = std::fs::read_to_string(file_name)?;


    let root = cfg.read(content)?;
    Ok(root)
}

pub(crate) fn new_ini(
    config: &HashMap<String, Option<String>>,
) -> Result<Ini, Box<dyn std::error::Error>> {
    let mut ini = Ini::new_cs();

    // Populate the DEFAULT section. Keys here are written with no section header.
    for (key, value) in config {
        // `set` takes Option<String>; None -> key with no value
        ini.set("default", key, value.clone());
    }

    // Optionally change an entry before writing:
    // ini.set("DEFAULT", "some_key", Some("new_value".into()));

    Ok(ini)
}