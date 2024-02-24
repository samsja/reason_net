use std::path::{Path, PathBuf};

use clap::Parser;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    min: u32,

    #[arg(long)]
    max: u32,

    #[arg(long)]
    size: u32,

    #[arg(long)]
    seed: u64,

    #[arg(long)]
    save_path: String,

    #[arg(long, default_value = "+-*/%")]
    operators: String,

    #[arg(long, default_value = "n")]
    short: String,

    #[arg(long, default_value = "40960")]
    chunk_size: usize,

    #[arg(long, default_value = "0.1")]
    val_prop: f32,
}

fn generate_operator(a: i64, b: i64, operator: char) -> String {
    match operator {
        '+' => format!("{a}+{b}={}", a + b),
        '-' => format!("{a}-{b}={}", a - b),
        '*' => format!("{a}*{b}={}", a * b),
        '/' => format!("{a}/{b}={}", a / b),
        '%' => format!("{a}%{b}={}", a % b),
        _ => panic!("Invalid operand"),
    }
}

fn generate_datapoint(min: u32, max: u32, rng: &mut StdRng, operators: String) -> String {
    let min_val = 10i64.pow(min);
    let max_val = 10i64.pow(max);

    let a: i64 = rng.gen_range(min_val..max_val);
    let b: i64 = rng.gen_range(min_val..max_val);

    let operand = rng.gen_range(0..operators.len());

    let operator = operators.chars().nth(operand).unwrap();

    let exo = generate_operator(a, b, operator);

    exo
}

fn creater_folder(path: &str) {
    match std::fs::create_dir(&Path::new(path)) {
        Ok(_) => {}
        Err(e) => println!("Failed to create directory: {}", e),
    }

    match std::fs::create_dir(&Path::new(path).join("val")) {
        Ok(_) => {}
        Err(e) => println!("Failed to create directory: {}", e),
    }

    match std::fs::create_dir(&Path::new(path).join("train")) {
        Ok(_) => {}
        Err(e) => println!("Failed to create directory: {}", e),
    }
}

fn generate_all(conf: Args) {
    creater_folder(&conf.save_path);

    let mut rng = StdRng::seed_from_u64(conf.seed);

    let mut unique_exos: FxHashSet<String> = FxHashSet::default();

    let progress_bar = ProgressBar::new(conf.size as u64);

    while unique_exos.len() < conf.size as usize {
        let datapoint: String =
            generate_datapoint(conf.min, conf.max, &mut rng, conf.operators.clone());
        let added = unique_exos.insert(datapoint);
        if added {
            progress_bar.inc(1);
        }
    }

    progress_bar.finish_with_message("done");

    let unique_exos: Vec<String> = unique_exos.into_iter().collect();
    let mut exo_to_save: Vec<String>;

    if conf.short == "y" {
        let operators: Vec<char> = conf.operators.chars().collect();

        let max = 999;

        let range1 = 9..max;
        let range2 = 9..max;

        let mut digit_data: Vec<String> = Vec::with_capacity(operators.len() * max * max);

        for operator in operators {
            for i in range1.clone() {
                for j in range2.clone() {
                    digit_data.push(generate_operator(i as i64, j as i64, operator))
                }
            }
        }

        exo_to_save = unique_exos
            .into_iter()
            .chain(digit_data.into_iter())
            .collect::<Vec<String>>();
    } else if conf.short == "n" {
        exo_to_save = unique_exos;
    } else {
        panic!("Invalid short option");
    }

    exo_to_save.shuffle(&mut rng);

    let split_index = (exo_to_save.len() as f64 * 0.9).floor() as usize;

    let exo_val = exo_to_save.split_off(split_index);

    let path_train = PathBuf::from(conf.save_path.clone()).join("train");
    for (i, chunk) in exo_to_save.chunks(conf.chunk_size).enumerate() {
        let path = path_train.join(format!("chunk_{}.txt", i));
        std::fs::write(path, chunk.join("\n")).unwrap();
    }

    let path_val = PathBuf::from(conf.save_path.clone()).join("val");
    for (i, chunk) in exo_val.chunks(conf.chunk_size).enumerate() {
        let path = path_val.join(format!("chunk_{}.txt", i));
        std::fs::write(path, chunk.join("\n")).unwrap();
    }
}

fn main() {
    let args = Args::parse();

    if args.min > args.max {
        panic!("min should be less than max");
    }

    if args.min < 3 {
        panic!("min should be greater than 3");
    }

    generate_all(args);
}
