use winit::{
    dpi::PhysicalSize,
    window::Window,
};
use thiserror::Error;

/// All the objects related to rendering including the device, command queue and surface
pub struct RenderState {
    /// The logical device connected to the gpu
    device: wgpu::Device,
    /// The command queue for sending info to the gpu
    queue: wgpu::Queue,
    /// The surface to draw on
    surface: wgpu::Surface,
    /// The configurations of the surface
    config: wgpu::SurfaceConfiguration,
}

impl RenderState {
    /// Creates a new render state from a given window
    /// 
    /// # Parameters
    /// 
    /// window: The window to use for the render state
    /// 
    /// # Errors
    /// 
    /// See NewRenderStateError for a description of the different errors which may occur
    pub async fn new(window: &Window) -> Result<Self, NewRenderStateError> {
        // Get the size of the window
        let size = window.inner_size();

        if size.width <= 0 || size.height <= 0 {
            return Err(NewRenderStateError::InvalidSize(size));
        }

        // Get a handle to the API
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::VALIDATION, // Any other choice crashes when running app in debug mode due to subtract with overflow error
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // Get a surface for the window
        let surface = unsafe { instance.create_surface(&window) }?;

        // Get an adapter to the GPU
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.ok_or(NewRenderStateError::GetAdapter)?;

        // Create a logical device and a command queue
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None,
        ).await?;
        
        // Get the capabilities of the surface
        let surface_caps = surface.get_capabilities(&adapter);

        // Get an sRGB texture format for the surface
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())            
            .ok_or(NewRenderStateError::IncompatibleSurface)?;

        // Setup the configurations and configure the surface
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface,
            config,
        })
    }

    /// Called when the window has been resized
    /// 
    /// # Parameters
    /// 
    /// new_size: The new size of the window
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Get a reference to the device
    pub fn get_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a mutable refence to the device
    pub fn get_device_mut(&mut self) -> &mut wgpu::Device {
        &mut self.device
    }

    /// The a reference to the queue
    pub fn get_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a mutable reference to the queue
    pub fn get_queue_mut(&mut self) -> &mut wgpu::Queue {
        &mut self.queue
    }

    /// Get a reference to the surface
    pub fn get_surface(&self) -> &wgpu::Surface {
        &self.surface
    }

    /// Get a mutable reference to the surface
    pub fn get_surface_mut(&mut self) -> &mut wgpu::Surface {
        &mut self.surface
    }

    /// Get a reference to the configs
    pub fn get_config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    /// Get a mutable reference to the configs
    pub fn get_config_mut(&mut self) -> &mut wgpu::SurfaceConfiguration {
        &mut self.config
    }
}

/// The error types for when creating a new RenderState
#[derive(Error, Debug, Clone)]
pub enum NewRenderStateError {
    /// Either the width or the height of the supplied window were to small
    #[error("The width and height of the window must be larger than 0 but received {:?}", .0)]
    InvalidSize(PhysicalSize<u32>),
    /// The surface could not be created
    #[error("Unable to create surface: {:?}", .0)]
    CreateSurface(wgpu::CreateSurfaceError),
    /// The gpu adapter could not be created
    #[error("Unable to get adapter for gpu")]
    GetAdapter,
    /// The device and queue could not be created
    #[error("Unable to retrieve logical device: {:?}", .0)]
    RequestDevice(wgpu::RequestDeviceError),
    /// There was no comatible surface on the device
    #[error("No compatible surface found")]
    IncompatibleSurface,
}

impl From<wgpu::CreateSurfaceError> for NewRenderStateError {
    fn from(value: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(value)
    }
}

impl From<wgpu::RequestDeviceError> for NewRenderStateError {
    fn from(value: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(value)
    }
}
