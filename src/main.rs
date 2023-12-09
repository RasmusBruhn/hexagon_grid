use hexagonal_grid as hex;
use std::env;

const N: usize = 10;
const COLOR_START: hex::color::Color = hex::color::Color::new_rgb(0.0, 0.0, 1.0);
const COLOR_END: hex::color::Color = hex::color::Color::new_rgb(0.0, 0.0, 0.0);
const FRAMERATE: f64 = 60.0;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // Create the color map
    let color_map = hex::color::ColorMap::new_linear(&COLOR_START, &COLOR_END, N + 1);

    // Run the application
    pollster::block_on(hex::application::run(hex::map::MapCyclic::<hex::map::TileID>::new_layered(N), color_map, FRAMERATE));
}
