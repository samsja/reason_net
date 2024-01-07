use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {

    #[arg(long)]
    min: u8,

    #[arg(long)]
    max: u8,

    #[arg(long)]
    size: u8,

    #[arg(long)]
    seed: u8,

    #[arg(long)]
    save_file_path: String,

}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
}