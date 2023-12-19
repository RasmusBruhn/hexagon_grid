use winit::{
    window::{Window, WindowBuilder},
    event_loop::{EventLoop, ControlFlow},
    dpi::PhysicalSize,
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState},
};
use thiserror::Error;
use std::time::{Instant, Duration};
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;
use super::{
    types::{Point, Transform2D},
    color::ColorMap,
    map,
    render::{RenderState, NewRenderStateError},
    gpu_map::{GPUMap, RenderError},
    camera::Camera,
};

/// Runs the application
/// 
/// # Parameters
/// 
/// map: The map used for this application
/// 
/// color_map: The coolor map for rendering
/// 
/// framerate: The wanted framerate
#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run<M: map::Map + 'static>(map: M, color_map: ColorMap, framerate: f64) {
    // Setup logging
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
    
    // Create the event loop
    let event_loop = EventLoop::new();

    // Create the window
    let size = PhysicalSize::new(500, 500);
    let window = WindowBuilder::new().with_inner_size(size).build(&event_loop);
    let window = match window {
        Ok(window) => window,
        Err(e) => {
            eprintln!("Unable to open window: {:?}", e);
            return;
        }
    };

    // Create canvas for browser to draw in
    #[cfg(target_arch = "wasm32")]
    {
        // Setup the canvas
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }    
    
    // Create the state
    let state = State::new(window, map, color_map, framerate).await;
    let mut state = match state {
        Ok(state) => state,
        Err(error) => {
            eprintln!("Unable to create state: {:?}", error);
            return;
        },
    };

    // Run the event loop
    event_loop.run(move |event, _, control_flow| state.handle_event(&event, control_flow));
}

/// Holds the state of the application
struct State<M: map::Map> {
    /// The main window
    window: Window,
    /// The render state
    render_state: RenderState,
    /// The camera
    camera: Camera,
    /// The inner size of the window
    size: PhysicalSize<u32>,
    /// The map of hexagons
    map: M,
    /// The gpu map
    gpu_map: GPUMap,
    /// The wanted framerate
    framerate: f64,
    /// The timestamp in milliseconds for the last frame
    last_time: Instant,
}

impl<M: map::Map> State<M> {
    /// Create a new state
    /// 
    /// # Parameters
    /// 
    /// window: The window to use for the application
    /// 
    /// map: The map to render
    /// 
    /// color_map: The color map for rendering
    /// 
    /// framerate: The wanted framerate
    /// 
    /// # Errors
    /// 
    /// See NewStateError for the possible errors
    async fn new(window: Window, map: M, color_map: ColorMap, framerate: f64) -> Result<Self, NewStateError> {
        // Get the size of the window
        let size = window.inner_size();

        if size.width <= 0 || size.height <= 0 {
            return Err(NewStateError::InvalidSize(size));
        }

        // Initialize the render state
        let render_state = RenderState::new(&window).await?;

        // Initialize the camera
        let camera = Camera::new(60.0, &Transform2D::scale(&Point::new(0.1, 0.1)), &size);

        // Initialize the gpu map
        let gpu_map = GPUMap::new(2.0, &camera.get_transform(), &color_map, &map, wgpu::include_wgsl!("shader.wgsl"), &render_state);

        // Get the current time
        let last_time = Instant::now();

        Ok (Self {
            window,
            render_state,
            camera,
            size,
            map,
            gpu_map,
            framerate,
            last_time,
        })
    }

    /// Render the screen
    /// 
    /// # Errors
    /// 
    /// See gpu_map::RenderError for the possible errors
    fn render(&self) -> Result<(), RenderError> {
        // Get the current view
        let output_texture = self.render_state.get_surface().get_current_texture()?;
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Draw the map
        self.gpu_map.draw(&view, &self.render_state)?;

        // Show to screen
        output_texture.present();

        Ok(())
    }

    /// Handles all events from winit
    /// 
    /// # Parameters
    /// 
    /// event: The event to handle
    /// 
    /// control_flow: The location to set the control flow
    fn handle_event(&mut self, event: &Event<'_, ()>, control_flow: &mut ControlFlow) {
        match event {
            // Run the window event handler
            Event::WindowEvent { window_id, event } => if *window_id == self.window.id() {
                self.handle_window_event(event, control_flow);
            }
    
            // Render the screen
            Event::RedrawRequested(window_id) => if *window_id == self.window.id() {
                match self.render() {
                    Ok(_) => {}
    
                    // Reconfigure the surface if lost
                    Err(RenderError::SurfaceTexture(wgpu::SurfaceError::Lost)) => self.resize(self.size),
    
                    // The system is out of memory, we should probably quit
                    Err(RenderError::SurfaceTexture(wgpu::SurfaceError::OutOfMemory)) => {
                        *control_flow = ControlFlow::Exit;
                        eprintln!("System is out of memory")
                    }
    
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("Error while rendering: {:?}", e),
                }
            }

            Event::MainEventsCleared => {
                // Update the camera
                if self.camera.update_transform() {
                    self.camera_changed()
                }

                // Limit the framerate
                let new_time = Instant::now();
                let wait_time = if 1e6 / self.framerate > (new_time.duration_since(self.last_time).as_micros() as f64) {
                    (1e6 / self.framerate - (new_time.duration_since(self.last_time).as_micros() as f64)) as u64
                } else {
                    0
                };
                *control_flow = ControlFlow::WaitUntil(new_time + Duration::from_micros(wait_time));
                self.last_time = new_time;
            }
    
            _ => ()
        }
    }

    /// Handle a window event
    /// 
    /// # Parameters
    /// 
    /// event: The event to handle
    /// 
    /// control_flow: The location to set the control flow
    fn handle_window_event(&mut self, event: &WindowEvent<'_>, control_flow: &mut ControlFlow) {
        match event {
            // Close the window
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
    
            // The size of the window has changed
            WindowEvent::Resized(physical_size) => {
                self.resize(*physical_size);
            }
    
            // The window has been dragged into an area with a different scale factor
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                self.resize(**new_inner_size);
            }

            WindowEvent::KeyboardInput { input, .. } => {
                self.handle_keyboard_input(input, control_flow);
            }

            _ => (),
        }
    }

    /// Handle a keyb oard input
    /// 
    /// # Parameters
    /// 
    /// input: The input to handle
    /// 
    /// control_flow: The location to set the control flow
    fn handle_keyboard_input(&mut self, input: &KeyboardInput, _control_flow: &mut ControlFlow) {
        // Get whether it is pressed
        let pressed = if let ElementState::Pressed = input.state {
            true
        } else {
            false
        };
        
        if let Some(keycode) = input.virtual_keycode {
            match keycode {
                VirtualKeyCode::D => self.camera.set_move(0, pressed),
                VirtualKeyCode::E => self.camera.set_move(1, pressed),
                VirtualKeyCode::W => self.camera.set_move(2, pressed),
                VirtualKeyCode::A => self.camera.set_move(3, pressed),
                VirtualKeyCode::Z => self.camera.set_move(4, pressed),
                VirtualKeyCode::X => self.camera.set_move(5, pressed),
                VirtualKeyCode::S => self.camera.set_zoom(0, pressed),
                VirtualKeyCode::Q => self.camera.set_zoom(1, pressed),
                VirtualKeyCode::R => self.camera.set_rotate(0, pressed),
                VirtualKeyCode::C => self.camera.set_rotate(1, pressed),
                _ => (),
            }    
        }
    }

    /// Reconfigure if the window has been resized
    /// 
    /// # Parameters
    /// 
    /// new_size: The new size to set
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Reconfigure the surface
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.render_state.resize(new_size);
        }

        // Update the camera
        self.camera.resize(&new_size);
        self.camera_changed();
    }

    /// Updates the rendering when the camera changed
    fn camera_changed(&mut self) {
        self.gpu_map.set_transform(&self.camera.get_transform(), &self.map, &self.render_state);
        self.window.request_redraw();
    }
}

/// The error types for when creating a new state
#[derive(Error, Debug, Clone)]
pub enum NewStateError {
    /// The width or height of the window is too small
    #[error("The width and height of the window must be larger than 0 but received {:?}", .0)]
    InvalidSize(PhysicalSize<u32>),
    /// The render state could not be created
    #[error("Unable to initialize the render state: {:?}", 0.)]
    RenderInitError(NewRenderStateError),
}

impl From<NewRenderStateError> for NewStateError {
    fn from(value: NewRenderStateError) -> Self {
        Self::RenderInitError(value)
    }
}