/// Describes a rgba color
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color {
    /// The red component
    pub r: f32,
    /// The green component
    pub g: f32,
    /// The blue component
    pub b: f32,
    /// The alpha component
    pub a: f32,
}

impl Color {
    /// Creates a new rgba color
    /// 
    /// # Parameters
    /// 
    /// r: The red component
    /// 
    /// g: The green component
    /// 
    /// b: The blue component
    /// 
    /// a: The alpha component
    pub const fn new_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
        }
    }

    /// Creates a new rgb color with a = 1
    /// 
    /// # Parameters
    /// 
    /// r: The red component
    /// 
    /// g: The green component
    /// 
    /// b: The blue component
    pub const fn new_rgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            r,
            g,
            b,
            a: 1.0,
        }
    }

    /// Creates a new gray scale color with a = 1, all other colors are equal
    /// 
    /// # Parameters
    /// 
    /// g: The value of all the color components
    pub const fn new_gray(g: f32) -> Self {
        Self {
            r: g,
            g: g,
            b: g,
            a: 1.0,
        }
    }
}

/// This describes a map between id and color values
#[derive(Clone, Debug)]
pub struct ColorMap(Vec<Color>);

impl ColorMap {
    /// Create a linear map such that id: 0 has color: start and id: size - 1 has color: end.
    /// All ids in between map to a linear interpolation of the 2 colors.
    /// 
    /// # Parameters
    /// 
    /// start: The starting color for id = 0
    /// 
    /// end: The ending color for id = size - 1
    /// 
    /// size: The number of id values, no id must later be given for which id >= size
    /// 
    /// # Panics
    /// 
    /// In debug mode this panics if size < 2
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

    /// Get the color data for writing to the gpu
    pub fn get_data(&self) -> &[Color] {
        &self.0
    }
}