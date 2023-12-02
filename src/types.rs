use std::ops::{Mul, Add, Sub};

/// A 2D point
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// The x-coordinate
    x: f64,
    /// The y-coordinate
    y: f64,
}

impl Point {
    /// Creates a new point
    /// 
    /// # Parameters
    /// 
    /// x: The x-coordinate
    /// 
    /// y: The y-coordinate
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
        }
    }

    /// Retrieves the x-coordinate
    pub fn get_x(&self) -> f64 {
        self.x
    }

    /// Retrieves the y-coordinate
    pub fn get_y(&self) -> f64 {
        self.y
    }
}

impl Add<Point> for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        let x = self.x + rhs.x;
        let y = self.y + rhs.y;

        Self {
            x,
            y,
        }
    }
}

impl Sub<Point> for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Self::Output {
        let x = self.x - rhs.x;
        let y = self.y - rhs.y;

        Self {
            x,
            y,
        }
    }
}

impl Mul<f64> for Point {
    type Output = Point;

    fn mul(self, rhs: f64) -> Self::Output {
        let x = self.x * rhs;
        let y = self.y * rhs;

        Self {
            x,
            y,
        }
    }
}

/// A 2D point
#[derive(Clone, Copy, Debug)]
pub struct Size {
    /// The width
    w: f64,
    /// The height
    h: f64,
}

impl Size {
    /// Creates a new size
    /// 
    /// # Parameters
    /// 
    /// w: The width
    /// 
    /// h: The height
    pub fn new(w: f64, h: f64) -> Self {
        let use_w = if w < 0.0 {
            -w
        } else {
            w
        };
        let use_h = if h < 0.0 {
            -h
        } else {
            h
        };

        Self {
            w: use_w,
            h: use_h,
        }
    }

    /// Retrieves the width
    pub fn get_w(&self) -> f64 {
        self.w
    }

    /// Retrieves the height
    pub fn get_h(&self) -> f64 {
        self.h
    }
}

impl Mul<f64> for Size {
    type Output = Size;

    fn mul(self, rhs: f64) -> Self::Output {
        let w = self.w * rhs;
        let h = self.h * rhs;
        Self {
            w,
            h,
        }
    }
}

impl Add<Size> for Size {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        let w = self.w + rhs.w;
        let h = self.h + rhs.h;

        Self {
            w,
            h,
        }
    }
}

/// A 2D index
#[derive(Clone, Copy, Debug)]
pub struct Index {
    /// The x-index
    x: i64,
    /// The y-index
    y: i64,
}

impl Index {
    /// Creates a new index
    /// 
    /// # Parameters
    /// 
    /// x: The x-index
    /// 
    /// y: The y-index
    pub fn new(x: i64, y: i64) -> Self {
        Self {
            x,
            y,
        }
    }

    /// Retrieves the x-index
    pub fn get_x(&self) -> i64 {
        self.x
    }

    /// Retrieves the y-index
    pub fn get_y(&self) -> i64 {
        self.y
    }
}

/// Defines a view of the map
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// The center of the rectangle
    center: Point,
    /// The size of the rectangle
    size: Size,
}

impl View {
    /// Creates a new view
    /// 
    /// # Parameters
    /// 
    /// center: The center of the rectangle
    /// 
    /// size: The size of the rectangle
    pub fn new(center: Point, size: Size) -> Self {
        Self {
            center,
            size,
        }
    }

    /// Retrieves the center
    pub fn get_center(&self) -> &Point {
        &self.center
    }

    /// Retrieves the size
    pub fn get_size(&self) -> &Size {
        &self.size
    }
}
