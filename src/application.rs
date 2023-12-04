use winit::{
    window::{Window, WindowBuilder},
    event_loop::{EventLoop, ControlFlow},
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
};
use thiserror::Error;
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use crate::types::{Point, Transform2D};

use super::{
    N,
    map,
    render::{RenderState, NewRenderStateError},
    gpu_map::{GPUMap, RenderError},
};

/// Runs the application
/// 
/// # Parameters
/// 
/// map: The map used for this application
#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run<M: map::Map + 'static>(map: M) {
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
    let state = State::new(window, map).await;
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
/*
/// Handles an event passed from winit
/// 
/// # Parameters
/// 
/// event: The window event that has occured, the window id should already have been tested\
/// control_flow: The control flow for the event loop\
/// state: The state of the application
fn event_handler<M: map::Map>(event: &Event<'_, ()>, control_flow: &mut ControlFlow, state: &mut RenderState, gpu_map: &mut GPUMap<M>) {
    match event {
        // Run the window event handler
        Event::WindowEvent { window_id, event } => if *window_id == state.window.id() {
            event_window_handler(event, control_flow, state);
        }

        // Render the screen
        Event::RedrawRequested(window_id) => if *window_id == state.window.id() {
            match state.render() {
                Ok(_) => {}

                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),

                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    *control_flow = ControlFlow::Exit;
                    eprintln!("System is out of memory")
                }

                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("Error while rendering: {:?}", e),
            }
        }

        _ => ()
    }
}

/// Handles a window event for the main window
/// 
/// # Parameters
/// 
/// event: The window event that has occured, the window id should already have been tested\
/// control_flow: The control flow for the event loop\
/// state: The state of the application
fn event_window_handler(event: &WindowEvent<'_>, control_flow: &mut ControlFlow, state: &mut State) {
    match event {
        // Close the window
        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

        // The size of the window has changed
        WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
        }

        // The window has been dragged into an area with a different scale factor
        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
            state.resize(**new_inner_size);
        }
        
        _ => (),
    }
}
*/
/// Holds the state of the application
struct State<M: map::Map> {
    /// The main window
    window: Window,
    /// The render state
    render_state: RenderState,
    /// The inner size of the window
    size: PhysicalSize<u32>,
    /// The map of hexagons
    _map: M,
    /// The gpu map
    gpu_map: GPUMap,
}

impl<M: map::Map> State<M> {
    /// Create a new state
    /// 
    /// # Parameters
    /// 
    /// window: The window to use for the application\
    /// map: The map to render
    async fn new(window: Window, map: M) -> Result<Self, NewStateError> {
        // Get the size of the window
        let size = window.inner_size();

        if size.width <= 0 || size.height <= 0 {
            return Err(NewStateError::InvalidSize(size));
        }

        // Initialize the render state
        let render_state = RenderState::new(&window).await?;

        // Initialize the gpu map
        let gpu_map = GPUMap::new(N, 2.0, &Transform2D::scale(&Point::new(0.1, 0.1)), &map, &render_state);

        Ok (Self {
            window,
            render_state,
            size,
            _map: map,
            gpu_map,
        })
    }

    /// Render the screen
    fn render(&self) -> Result<(), RenderError> {
        self.gpu_map.render(&self.render_state)
    }

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
    
            _ => ()
        }
    }

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
            
            _ => (),
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
    }
}

#[derive(Error, Debug, Clone)]
pub enum NewStateError {
    #[error("The width and height of the window must be larger than 0 but received {:?}", .0)]
    InvalidSize(PhysicalSize<u32>),
    #[error("Unable to initialize the render state: {:?}", 0.)]
    RenderInitError(NewRenderStateError),
}

impl From<NewRenderStateError> for NewStateError {
    fn from(value: NewRenderStateError) -> Self {
        Self::RenderInitError(value)
    }
}