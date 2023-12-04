const N: usize = 4;
const SQRT_3: f64 = 1.7320508075688772935274463415058723669428052538103806280558069794;
const INV_SQRT_3: f64 = 0.5773502691896257645091487805019574556476017512701268760186023264;
const START_COLOR: [f32; 4] = [ 0.0, 0.0, 1.0, 1.0 ];
const END_COLOR: [f32; 4] = [ 0.0, 0.0, 0.0, 1.0 ];

pub mod map;
pub mod gpu_map;
pub mod application;
pub mod types;
pub mod render;
