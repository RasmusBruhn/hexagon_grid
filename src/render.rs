use winit::{
    dpi::PhysicalSize,
    window::Window,
};
use thiserror::Error;

pub struct RenderState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
}

impl RenderState {
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

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn get_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn get_device_mut(&mut self) -> &mut wgpu::Device {
        &mut self.device
    }

    pub fn get_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn get_queue_mut(&mut self) -> &mut wgpu::Queue {
        &mut self.queue
    }

    pub fn get_surface(&self) -> &wgpu::Surface {
        &self.surface
    }

    pub fn get_surface_mut(&mut self) -> &mut wgpu::Surface {
        &mut self.surface
    }

    pub fn get_config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    pub fn get_config_mut(&mut self) -> &mut wgpu::SurfaceConfiguration {
        &mut self.config
    }
}

#[derive(Error, Debug, Clone)]
pub enum NewRenderStateError {
    #[error("The width and height of the window must be larger than 0 but received {:?}", .0)]
    InvalidSize(PhysicalSize<u32>),
    #[error("Unable to create surface: {:?}", .0)]
    CreateSurface(wgpu::CreateSurfaceError),
    #[error("Unable to get adapter for gpu")]
    GetAdapter,
    #[error("Unable to retrieve logical device: {:?}", .0)]
    RequestDevice(wgpu::RequestDeviceError),
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
