use super::{
    SQRT_3,
    types::{Transform2D, Point}
};

/// Describes a how the camera is moving
pub struct Camera {
    /// The movement keys: d, e, w, a, z, x
    move_active: [bool; 6],
    /// The zoom keys: s, q
    zoom_active: [bool; 2],
    /// The rotation keys: r, c
    rotate_active: [bool; 2],
    /// True if any button is pressed and the camera needs to be updated
    active: bool,
    /// The framerate of the program, this is how many times a second the transform should be updated
    framerate: f64,
    /// The current transform
    transform: Transform2D,
    /// The transform to make the aspect ratio correct
    transform_aspect: Transform2D,
    /// The transform to apply to the current transform every frame
    transform_update: Transform2D,
}

impl Camera {
    /// Creates a new camera
    /// 
    /// # Parameters
    /// 
    /// framerate: The expected framerate of the program, this is how many times a second the transform should be updated
    /// 
    /// transform: The initial transform to use
    /// 
    /// size: The current size of the window
    pub fn new(framerate: f64, transform: &Transform2D, size: &winit::dpi::PhysicalSize<u32>) -> Self {
        Self {
            move_active: [false; 6],
            zoom_active: [false; 2],
            rotate_active: [false; 2],
            active: false,
            framerate,
            transform: *transform,
            transform_aspect: Self::size_to_aspect(size),
            transform_update: Transform2D::identity(),
        }
    }

    /// Set one of the movement keys, id 0-5 for d, e, w, a, z, x
    /// 
    /// # Parameters
    /// 
    /// id: The id of the key to set
    /// 
    /// active: True if it is pressed down
    pub fn set_move(&mut self, id: usize, active: bool) {
        self.move_active[id] = active;
        self.reload_transform();
    }

    /// Set one of the zoom keys, id 0-1 for s, q
    /// 
    /// # Parameters
    /// 
    /// id: The id of the key to set
    /// 
    /// active: True if it is pressed down
    pub fn set_zoom(&mut self, id: usize, active: bool) {
        self.zoom_active[id] = active;
        self.reload_transform();
    }

    /// Set one of the rotate keys, id 0-1 for r, c
    /// 
    /// # Parameters
    /// 
    /// id: The id of the key to set
    /// 
    /// active: True if it is pressed down
    pub fn set_rotate(&mut self, id: usize, active: bool) {
        self.rotate_active[id] = active;
        self.reload_transform();
    }

    /// Reset all of the input such that all of it is turned off
    pub fn reset_updates(&mut self) {
        self.move_active.iter_mut().for_each(|val| *val = false);
        self.zoom_active.iter_mut().for_each(|val| *val = false);
        self.rotate_active.iter_mut().for_each(|val| *val = false);
        self.reload_transform();
    }

    /// Sets the framerate for if it changes
    /// 
    /// # Parameters
    /// 
    /// framerate: The new framerate, this is how many times a second the transform should be updated
    pub fn set_framerate(&mut self, framerate: f64) {
        self.framerate = framerate;
        self.reload_transform();
    }

    /// Recalculates the aspect transform after resizing
    /// 
    /// # Parameters
    /// 
    /// size: THe new size of the window
    pub fn resize(&mut self, size: &winit::dpi::PhysicalSize<u32>) {
        self.transform_aspect = Self::size_to_aspect(size);
    }

    /// Retrieves the transform
    pub fn get_transform(&self) -> Transform2D {
        &self.transform_aspect * self.transform
    }

    /// Sets a new transform
    /// 
    /// # Parameters
    /// 
    /// transform: The new transform to set
    pub fn set_transform(&mut self, transform: &Transform2D) {
        self.transform = *transform;
    }

    /// Update the transform using the current input, should be run once per frame
    /// 
    /// Returns true if the transform has updated
    pub fn update_transform(&mut self) -> bool {
        if !self.active {
            return false;
        }

        self.transform = self.transform_update * self.transform;

        true
    }

    /// Reload the transform_update for when the input has changed
    fn reload_transform(&mut self) {
        // Check if it is active
        self.active = self.move_active.iter().any(|&x| x) || self.zoom_active.iter().any(|&x| x) || self.rotate_active.iter().any(|&x| x);

        if !self.active {
            return;
        }

        // Calculate the movement direction
        let move_val = 4.0 / self.framerate;
        let key_move = [
            Point::new(1.0, 0.0),
            Point::new(0.5, -0.5 * SQRT_3),
            Point::new(-0.5, -0.5 * SQRT_3),
            Point::new(-1.0, 0.0),
            Point::new(-0.5, 0.5 * SQRT_3),
            Point::new(0.5, 0.5 * SQRT_3),
        ];
        let mut move_dir = self.move_active
            .iter()
            .zip(key_move.iter())
            .filter_map(|(&active, dir)| {
                if active {
                    Some(dir)
                } else {
                    None
                }
            })
            .fold(Point::new(0.0, 0.0), |prev, next| prev + next);
        if move_dir.get_x() != 0.0 || move_dir.get_y() != 0.0 {
            move_dir = move_dir * (move_val / (move_dir.get_x() * move_dir.get_x() + move_dir.get_y() * move_dir.get_y()).sqrt());
        }

        // Calculate the zoom direction
        let zoom_val = 1.0 + 1.2 / self.framerate;
        let key_zoom = [
            zoom_val,
            1.0 / zoom_val,
        ];
        let zoom_dir = self.zoom_active
            .iter()
            .zip(key_zoom.iter())
            .filter_map(|(&active, zoom)| {
                if active {
                    Some(zoom)
                } else {
                    None
                }
            })
            .fold(1.0, |prev, next| prev * next);

        // Calculate the rotation direction
        let rotate_val = 1.0 / self.framerate;
        let key_rotate = [
            rotate_val,
            -rotate_val,
        ];
        let rotate_dir = self.rotate_active
            .iter()
            .zip(key_rotate.iter())
            .filter_map(|(&active, zoom)| {
                if active {
                    Some(zoom)
                } else {
                    None
                }
            })
            .fold(0.0, |prev, next| prev + next);

        // Combine all of the transforms
        let transform_move = Transform2D::translate(&move_dir);
        let transform_zoom = Transform2D::scale(&Point::new(zoom_dir, zoom_dir));
        let transform_rotate = Transform2D::rotation(rotate_dir);

        self.transform_update = transform_rotate * transform_zoom * transform_move;
    }

    /// Converts a size to an aspect transform
    /// 
    /// # Parameters
    /// 
    /// size: The size of the window
    fn size_to_aspect(size: &winit::dpi::PhysicalSize<u32>) -> Transform2D {
        Transform2D::scale(&Point::new((size.height as f64) / (size.width as f64), -1.0))
    }
}