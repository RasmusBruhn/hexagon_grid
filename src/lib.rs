pub mod map;
pub mod gpu_map;
pub mod application;
pub mod types;
pub mod render;
pub mod color;

const N: usize = 4;
const SQRT_3: f64 = 1.7320508075688772935274463415058723669428052538103806280558069794;
const INV_SQRT_3: f64 = 0.5773502691896257645091487805019574556476017512701268760186023264;
const COLOR_START: color::Color = color::Color::new_rgb(0.0, 0.0, 1.0);
const COLOR_END: color::Color = color::Color::new_rgb(0.0, 0.0, 0.0);
