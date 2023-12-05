use super::{
    SQRT_3,
    types::{Transform2D, Point}
};

/// Describes a how the camera is moving
pub struct Camera {
    /// The movement keys: e, w, a, z, x, d
    pub move_active: [bool; 6],
    /// The zoom keys: s, q
    pub zoom_active: [bool; 2],
    /// The rotation keys: r, c
    pub rotation_active: [bool; 2],
}

impl Camera {
    pub fn new() -> Self {
        Self {
            move_active: [false; 6],
            zoom_active: [false; 2],
            rotation_active: [false; 2],
        }
    }

    pub fn transform(&self, framerate: f64) -> Transform2D {
        // Calculate the movement direction
        let key_dir = [
            Point::new(0.5, 0.5 * SQRT_3),
            Point::new(-0.5, 0.5 * SQRT_3),
            Point::new(-1.0, 0.0),
            Point::new(-0.5, -0.5 * SQRT_3),
            Point::new(0.5, -0.5 * SQRT_3),
            Point::new(1.0, 0.0)
        ];
        let mut move_dir = self.move_active
            .iter()
            .zip(key_dir.iter())
            .filter_map(|(&active, dir)| {
                if active {
                    Some(dir)
                } else {
                    None
                }
            })
            .fold(Point::new(0.0, 0.0), |prev, next| prev + next);
        if move_dir.get_x() != 0.0 || move_dir.get_y() != 0.0 {
            move_dir = move_dir * (1.0 / (move_dir.get_x() * move_dir.get_x() + move_dir.get_y() * move_dir.get_y()).sqrt());
        }

        // Calculate the zoom
        let zoom_val = 2.0;
        let zoom_dir = 
    }
}