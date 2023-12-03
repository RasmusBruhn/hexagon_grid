use std::ops::{Mul, Add, Sub, Neg};

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

    /// Retrieves the data for the gpu
    pub fn get_data(&self) -> [f32; 2] {
        [self.x as f32, self.y as f32]
    }
}

impl Neg for Point {
    type Output = Point;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
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

/// Defines a 2x2 matrix
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    values: [[f64; 2]; 2],
}

impl Matrix {
    /// Creates a new matrix
    /// 
    /// # Parameters
    /// 
    /// values: The values of the matrix, first index is row, second index is column
    pub fn new(values: &[[f64; 2]; 2]) -> Self {
        Self { values: *values }
    }

    /// Transposes the matrix
    pub fn transpose(&self) -> Self {
        Self::new(&[[self.values[0][0], self.values[1][0]], [self.values[0][1], self.values[1][1]]])
    }

    /// Inverts the matrix
    /// 
    /// # Panics
    /// 
    /// In debug mode it panics if the determinant is 0 (it is not invertible)
    pub fn inv(&self) -> Self {
        // Calculate determinant
        let d = self.values[0][0] * self.values[1][1] - self.values[0][1] * self.values[1][0];

        // Make sure it is not invalid
        if cfg!(debug_assertions) && d == 0.0 {
            panic!("The matrix is not invertible: {:?}", self);
        }

        // Calculate inverse
        Self::new(&[[self.values[1][1], -self.values[0][1]], [-self.values[1][0], self.values[0][0]]])
    }

    /// Retrieves the data for the gpu
    pub fn get_data(&self) -> [f32; 4] {
        [self.values[0][0] as f32, self.values[0][1] as f32, self.values[1][0] as f32, self.values[1][1] as f32]
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        Self::new(&[[
                self.values[0][0] * rhs.values[0][0] + self.values[0][1] * rhs.values[1][0],
                self.values[0][0] * rhs.values[0][1] + self.values[0][1] * rhs.values[1][1]
            ], [
                self.values[1][0] * rhs.values[0][0] + self.values[1][1] * rhs.values[1][0],
                self.values[1][0] * rhs.values[0][1] + self.values[1][1] * rhs.values[1][1],
        ]])
    }
}

impl Neg for Matrix {
    type Output = Matrix;

    fn neg(self) -> Self::Output {
        Self::new(&[[-self.values[0][0], -self.values[0][1]], [-self.values[1][0], -self.values[1][1]]])
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        Self::new(&[[
                self.values[0][0] + rhs.values[0][0],
                self.values[0][1] + rhs.values[0][1],
            ], [
                self.values[1][0] + rhs.values[1][0],
                self.values[1][1] + rhs.values[1][1],
        ]])
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        Self::new(&[[
                self.values[0][0] - rhs.values[0][0],
                self.values[0][1] - rhs.values[0][1],
            ], [
                self.values[1][0] - rhs.values[1][0],
                self.values[1][1] - rhs.values[1][1],
        ]])
    }
}

impl Mul<Point> for Matrix {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        Point::new(
            self.values[0][0] * rhs.x + self.values[0][1] * rhs.y,
            self.values[1][0] * rhs.x + self.values[1][1] * rhs.y,
        )
    }
}

/// A 2D transform which acts on Point types, including rotation, scaling and translation
#[derive(Copy, Clone, Debug)]
pub struct Transform2D {
    center_transform: Matrix,
    center: Point,
}

impl Transform2D {
    /// Creates the identity operation
    pub fn identity() -> Self {
        let center_transform = Matrix::new(&[[1.0, 0.0], [0.0, 1.0]]);
        let center = Point::new(0.0, 0.0);

        Self { center_transform, center }
    }

    /// Rotate around origo
    /// 
    /// # Parameters
    /// 
    /// angle: The angle to rotate
    pub fn rotation(angle: f64) -> Self {
        let center_transform = Matrix::new(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
        let center = Point::new(0.0, 0.0);

        Self { center_transform, center }
    }
    // r * (x - c) + c = r * (x - c + r^-1 * c)
    /// Rotate around center
    /// 
    /// # Parameters
    /// 
    /// angle: The angle to rotate
    /// 
    /// rotation_center: The center of the rotation
    pub fn rotation_at(angle: f64, rotation_center: &Point) -> Self {
        let center_transform = Matrix::new(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
        let center = *rotation_center - center_transform.inv() * *rotation_center;

        Self { center_transform, center }
    }

    /// Scale at origo
    /// 
    /// # Parameters
    /// 
    /// scale: The ratio to scale x and y with
    pub fn scale(scale: &Point) -> Self {
        let center_transform = Matrix::new(&[[scale.x, 0.0], [0.0, scale.y]]);
        let center = Point::new(0.0, 0.0);

        Self { center_transform, center }
    }

    /// Scale at center
    /// 
    /// # Parameters
    /// 
    /// scale: The ratio to scale x and y with
    /// 
    /// center: The center of the scaling
    pub fn scale_at(scale: &Point, scale_center: &Point) -> Self {
        let center_transform = Matrix::new(&[[scale.x, 0.0], [0.0, scale.y]]);
        let center = *scale_center - center_transform.inv() * *scale_center;

        Self { center_transform, center }
    }

    /// Translates a point
    /// 
    /// # Parameters
    /// 
    /// offset: The amount to translate
    pub fn translate(offset: &Point) -> Self {
        let center_transform = Matrix::new(&[[1.0, 0.0], [0.0, 1.0]]);
        let center = *offset;

        Self { center_transform, center }
    }

    /// Retrieves the inverse transform
    pub fn inv(&self) -> Self {
        let center_transform = self.center_transform.inv();
        let center = -self.center_transform * self.center;

        Self { center_transform, center }
    }

    /// Retrieves the offset
    pub fn get_center(&self) -> Point {
        self.center
    }

    /// Retrieves the center transform
    pub fn get_center_transform(&self) -> Matrix {
        self.center_transform
    }

    /// Retrieves the data for the offset
    pub fn get_data_offset(&self) -> [f32; 2] {
        self.center.get_data()
    }

    /// Retrieves the data for the center transform
    pub fn get_data_center_transform(&self) -> [f32; 4] {
        self.center_transform.get_data()
    }
}

impl Mul<Transform2D> for Transform2D {
    type Output = Transform2D;

    /// t2 * t1 * x = r2 * (r1 * (x - c1) - c2) = r2 * r1 * (x - c1 - r1^-1 * c2)
    fn mul(self, rhs: Transform2D) -> Self::Output {
        let center_transform = self.center_transform * rhs.center_transform;
        let center = rhs.center + rhs.center_transform.inv() * self.center;

        Self { center_transform, center }
    }
}

impl Mul<Point> for Transform2D {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        self.center_transform * (rhs - self.center)
    }
}
