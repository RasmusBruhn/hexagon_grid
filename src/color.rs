#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
        }
    }

    pub const fn new_rgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            r,
            g,
            b,
            a: 1.0,
        }
    }

    pub const fn new_gray(g: f32) -> Self {
        Self {
            r: g,
            g: g,
            b: g,
            a: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColorMap(Vec<Color>);

impl ColorMap {
    pub fn new_linear(start: &Color, end: &Color, size: usize) -> Self {
        // Make sure size is large enough
        if cfg!(debug_assertions) && size < 2 {
            panic!("size must be at least 2");
        }

        let map = (0..size)
            .map(|id| {
                let fraction = (id as f32) / ((size - 1) as f32);
                Color {
                    r: fraction * (end.r - start.r) + start.r,
                    g: fraction * (end.g - start.g) + start.g,
                    b: fraction * (end.b - start.b) + start.b,
                    a: fraction * (end.a - start.a) + start.a,
                }
            })
            .collect();
        
        Self(map)
    }

    pub fn get_data(&self) -> &[Color] {
        &self.0
    }
}