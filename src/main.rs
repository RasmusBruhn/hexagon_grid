use hexagonal_grid as hex;
use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // Run the application
    pollster::block_on(hex::application::run(hex::map::MapCyclic::<hex::map::TileTest>::new_layered()));
}
